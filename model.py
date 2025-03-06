import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import torch.optim as optim
import torch
import copy 
import inspect

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=11, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Transformer(nn.Module):
    def __init__(self, num_layers, decoder_layer):
        super().__init__()
        self.layers = clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt):
        output = tgt

        for mod in self.layers:
            output = mod(output)
        return output
    
class LinearTokenizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = nn.Conv3d(
            in_channels=1,  # Input channels (C=1)
            out_channels=config.n_embd,
            kernel_size=config.token_width,
            stride=config.token_width,
            padding=0,
            bias=True  # Optional
        )

        
    def forward(self, x):
        # x shape: (B, 1, D=96, H=96, W=96)
        x = self.tokenizer(x)  # Output shape: (B, n_embd, D_blocks=8, H_blocks=8, W_blocks=8)
        
        # Efficient reshaping and transposing
        x = x.reshape(x.size(0), x.size(1), -1).transpose(-2, -1)  # (B, 512, n_embd)
        
        return x


@dataclass
class LinearConfig:
    block_size: int = int((96**3)/(6**3)) # max sequence length
    token_width: int = 6 # width of the cube
    n_layer: int = 6 # number of layers
    n_head: int = 4 # number of heads
    n_embd: int = 128 # embedding dimension
    n_class: int = 7 # numnber of classes

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = LinearTokenizer(config)
        self.positional_embedding = nn.Parameter(torch.zeros(config.block_size, config.n_embd))
        self.transformer = Transformer(config.n_layer, TransformerBlock(config))
        self.decoder = nn.Linear(config.n_embd, config.n_class)  # Output layer

    def ff(self, x):
        x = self.tokenizer(x) # (N, 1, 96, 96, 96) -> (N, 8 * 8 * 8, n_embd)
        x = x + self.positional_embedding # (N, n_embd, 8, 8, 8) -> (N, n_embd, 512) -> (N, 512, n_embd)
        x = self.transformer(x) # (N, 512, n_embd) -> (N, 512, n_embd)
        x = self.decoder(x) # (N, 512, n_embd) -> (N, 512, 7)        
        return x
    
    def forward(self, x):   
        return self.ff(x)
    

class ContrastiveModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Summarizer token
        self.summarizer = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        self.projection_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.ReLU(),
            nn.Linear(config.n_embd, 128)  # Project to lower-dimensional space
        )

    def forward(self, x):
        x = self.tokenizer(x) # (N, 1, 96, 96, 96) -> (N, 8 * 8 * 8, n_embd)
        x = x + self.positional_embedding # (N, n_embd, 8, 8, 8) -> (N, n_embd, n_tokens) -> (N, n_tokens, n_embd)
        
        # Append Summarizer token
        x = torch.cat([self.summarizer.repeat(x.size(0), 1, 1), x], dim=1)
        
        # Build representation using transformer
        x = self.transformer(x) # (N, n_tokens + 1, n_embd) -> (N, n_tokens + 1, n_embd)
        
        # Extract the summarizer token
        x = x[:, 0, :]  
        # Project to contrastive space
        return self.projection_head(x)
    
class SegmentationModel(ContrastiveModel):
    def __init__(self, config):
        super().__init__(config)
    
    def ff_without_summarizer(self, x):
        # ContrastiveModel inherits the ff (feedforward without summarizer) method from the BaseModel
        return self.ff(x)
    
    def ff_with_summarizer(self, x):
        x = self.tokenizer(x) # (N, 1, 96, 96, 96) -> (N, 8 * 8 * 8, n_embd)
        x = x + self.positional_embedding # (N, n_embd, 8, 8, 8) -> (N, n_embd, n_tokens) -> (N, n_tokens, n_embd)
        
        # Append Summarizer token
        x = torch.cat([self.summarizer.repeat(x.size(0), 1, 1), x], dim=1)
        
        # Build representation using transformer
        x = self.transformer(x) # (N, n_tokens + 1, n_embd) -> (N, n_tokens + 1, n_embd)     

        # Classify
        x = self.decoder(x[:, 1:, :])
        return x       

    def forward(self, x):
        return self.ff_without_summarizer(x)


    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    