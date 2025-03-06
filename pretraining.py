# Library imports
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
import json
import os
from rich import print as rprint  # Import rich's print function
import copy
from dataclasses import dataclass
import time
import inspect
from utils import *
# set torch and cuda seed for reproducibility
torch.manual_seed(37)
torch.cuda.manual_seed(37)
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------DATASET IMPLEMENTATION-----------------#
class TomogramDatasetMiniCubes(Dataset):
    def __init__(self, tomogram_data, segmentation_labels):
        assert tomogram_data.size(0) == segmentation_labels.size(0), f"{tomogram_data.size(0)}, {segmentation_labels.size(0)}"
        self.tomogram_data = tomogram_data
        self.segmentation_labels = segmentation_labels

    def __len__(self):
        return self.tomogram_data.size(0)

    def __getitem__(self, idx):
        return (
            self.tomogram_data[idx],
            self.segmentation_labels[idx],
        )


input_data, segmentation_labels = torch.load("segmentation_input_data.pt"), torch.load("segmentation_labels.pt")
# Create the dataset
particle_dataset_mini_cubes = TomogramDatasetMiniCubes(input_data, segmentation_labels)

labels_tensor = get_all_labels(particle_dataset_mini_cubes)
plot_label_distribution_torch(labels_tensor)
inv_freq_class_weights = calculate_class_weights(labels_tensor, 7, device)
# print(f"Inv Freq Weights: {inv_freq_class_weights}")


# Test the dataset
cube_data, segmentation_labels = particle_dataset_mini_cubes[200]
print(f"len(particle_dataset_mini_cubes): {len(particle_dataset_mini_cubes)}")
print("Cube Data Shape:", cube_data.shape)  # Should be (1, 96, 96, 96)
print("Labels Shape:", segmentation_labels.shape)        # Should be (4096,)

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
    n_embd: int = 64 # embedding dimension
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
    

torch.set_float32_matmul_precision("high")


config = LinearConfig()
segmentation_model = SegmentationModel(config).to(device)
segmentation_model = torch.compile(segmentation_model)
learning_rate = 1e-3


# Define optimizer, loss, and scheduler
optimizer = segmentation_model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate, device=device)
print(f"learning rate: {learning_rate}")

try:
    # Load the trained model
    # segmentation_model.load_state_dict(torch.load("segmentation_model_mini_cubes.pth"))
    # print("Loaded pretrained model")
    pass
except:
    pass
# print the # of parameters
print(f"Number of parameters: {sum(p.numel() for p in segmentation_model.parameters() if p.requires_grad):.2e}")
print(segmentation_model)
# Write train and validation dataloader for segmentation using particle_dataset_mini_cubes
train_size = int(0.8 * len(particle_dataset_mini_cubes))
val_size = len(particle_dataset_mini_cubes) - train_size
print(f"Train Size: {train_size}, Val Size: {val_size}")
train_dataset, val_dataset = random_split(particle_dataset_mini_cubes, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

"""
particle_types = {"virus-like-particle": 1, "apo-ferritin": 2, "beta-amylase": 3, 
                  "beta-galactosidase": 4, "ribosome": 5, "thyroglobulin": 6}
"""
# background, virus-like particle, "apo-ferritin", "beta-amylase","beta-galactosidase","ribosome","thyroglobulin"
fbeta_weights = torch.tensor([0, 1, 1, 0, 2, 1, 2]).to(device)
print(f"beta weights: {fbeta_weights}")

weights = torch.tensor([1, 1, 1, 1, 2, 1, 2]).to(device) * inv_freq_class_weights
print(f"Weights: {weights}")

criterion = nn.CrossEntropyLoss(weight=weights)

# Training loop
epochs = 200
best_val_loss = float('inf')
val_fb4_scores = []

batch_0 = next(iter(train_loader))
for epoch in range(epochs):
    segmentation_model.train()
    train_loss = 0.0
    train_fb4_scores = []
    # print(f"Epoch: {epoch + 1}")
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Measure time to load batch
        # st = time.time()
        # torch.cuda.synchronize()
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            outputs = segmentation_model(inputs)
            outputs, labels = outputs.view(-1, outputs.size(-1)), labels.view(-1)
            # loss = criterion(outputs, labels)
            loss = weighted_tversky_loss(pred = outputs, target = labels, class_weights = weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(segmentation_model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
        fb4_score = compute_fbeta_loss(torch.argmax(outputs, -1), labels, fbeta_weights, segmentation_model.config.n_class, device)
        train_fb4_scores.append(fb4_score)
        train_recall = compute_recall_per_class(torch.argmax(outputs.detach(), -1), labels.detach(), segmentation_model.config.n_class)
        # print(f"Iter: {str(batch_idx).zfill(3)} | Loss: {loss.item():.6f} | Recall: {train_recall}, F4-Beta: {fb4_score.item():.4f}")
        # torch.cuda.synchronize()
        # et = time.time()
        # print(f"{(inputs.shape[0] * labels.shape[-1]) / (et - st):.6g} tokens/s")
    if epoch == (epochs // 2):
        for param_group in optimizer.param_groups:
            param_group['lr'] = (learning_rate / 10)

    epoch_mean_fb4_score = torch.mean(torch.cat(train_fb4_scores)).item()
    # print(f"Epoch: {str(epoch + 1).zfill(3)}, Train Loss: {train_loss / len(train_loader):.6f}, fb4_score: {epoch_mean_fb4_score:.4f}")

    val_loss = 0.0
    segmentation_model.eval()
    with torch.no_grad():
        recall_out, recall_labels = [], []
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = segmentation_model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            val_loss += loss.item()
            recall_out.append(outputs)
            recall_labels.append(labels)
        # Compute Recall
        recall_out = torch.cat(recall_out)
        recall_labels = torch.cat(recall_labels)
        rout, rlabels = recall_out.view(-1, recall_out.size(-1)), recall_labels.view(-1)
        fb4_score = compute_fbeta_loss(torch.argmax(rout, -1), rlabels, fbeta_weights, segmentation_model.config.n_class, device)
        recall = compute_recall_per_class(torch.argmax(rout, -1), rlabels, segmentation_model.config.n_class)
            
    # Log epoch performance
    print(f"Epoch [{epoch + 1}/{epochs}] Train Loss: {train_loss / len(train_loader):.6f} Train Fb4: {epoch_mean_fb4_score:.4f}, Validation Loss: {val_loss / len(val_loader):.6f}, Recall: {recall}, F4-Beta: {fb4_score.item():.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(segmentation_model.state_dict(), "segmentation_model_mini_cubes.pth")
        print(f"#####------saved model at Epoch {epoch + 1}-----------#")    
