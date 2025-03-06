# Library imports
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
import zarr
import json
import os
import napari
from rich import print as rprint  # Import rich's print function
import copy
from dataclasses import dataclass

# set torch and cuda seed for reproducibility
torch.manual_seed(37)
torch.cuda.manual_seed(37)

VISUALIZE = False
#-------------LOADING TOMOGRAM DATA AND PARTICLE COORDINATES-----------------#

# Define the experiment runs to load
experiment_runs = ["TS_5_4", "TS_69_2", "TS_6_4", "TS_6_6", "TS_73_6", "TS_86_3", "TS_99_9"]
particle_types = {"virus-like-particle":1, "apo-ferritin":2, "beta-amylase":3, "beta-galactosidase":4, "ribosome":5, "thyroglobulin":6}
voxel_spacing = [10.0, 10.0, 10.0]  # 10 angstroms per voxel

# Initialize lists to store combined data
combined_tomogram_data = []
combined_particle_coords = {pt: [] for pt in particle_types}

# Track the cumulative z-depth for coordinate translation
cumulative_z_depth = 0

# Load and combine data from all experiment runs
for experiment_run in experiment_runs:
    zarr_file_path = os.path.join("train", "static", "ExperimentRuns", experiment_run, "VoxelSpacing10.000", "denoised.zarr")
    json_base_path = os.path.join("train", "overlay", "ExperimentRuns", experiment_run, "Picks")

    # Load the Zarr file
    try:
        tomogram = zarr.open(zarr_file_path, mode="r")
        tomogram_data = tomogram["0"][:]  # Load into memory as a NumPy array
        print(f"Tomogram shape for {experiment_run} (z, y, x):", tomogram_data.shape)
        tomogram_data = (tomogram_data - tomogram_data.mean()) / tomogram_data.std()
        combined_tomogram_data.append(tomogram_data)
    except Exception as e:
        print(f"Error loading Zarr file for {experiment_run}: {e}")
        continue

    # Load and transform particle coordinates for all types
    for particle_type in particle_types:
        json_file_path = os.path.join(json_base_path, f"{particle_type}.json")
        try:
            with open(json_file_path, "r") as file:
                data = json.load(file)
            points = data["points"]

            # Convert from real-world coordinates (angstroms) to voxel indices and reorder to (z, y, x)
            coords = np.array([
                [
                    (p["location"]["z"] / voxel_spacing[0]) + cumulative_z_depth,  # Translate z-coordinate
                     p["location"]["y"] / voxel_spacing[1],  # y-coordinate
                     p["location"]["x"] / voxel_spacing[2],  # x-coordinate
                ]
                for p in points
            ])
            combined_particle_coords[particle_type].extend(coords)
            print(f"Loaded {len(coords)} points for {particle_type} in {experiment_run}.")
        except Exception as e:
            print(f"Error loading JSON file for {particle_type} in {experiment_run}: {e}")

    # Update cumulative_z_depth for the next tomogram
    cumulative_z_depth += tomogram_data.shape[0]

# Combine all tomogram data into a single array
combined_tomogram_data = np.concatenate(combined_tomogram_data, axis=0)
print("Combined tomogram shape (z, y, x):", combined_tomogram_data.shape)

# Print total number of particles
total_particles = sum(len(coords) for coords in combined_particle_coords.values())
print(f"Total number of particles: {total_particles}")

# --------------------------------------------------------------------------------------------#
#-------------Combine tomograms and sample cubes with particles in it-----------------#

# Dimensions of the combined tomogram data
data_shape = combined_tomogram_data.shape
cube_size = (96, 96, 96)
particle_label_size = (8, 8, 8)
background_id = 0

# Calculate the number of cubes in each dimension
num_cubes_z = data_shape[0] // cube_size[0]
num_cubes_y = data_shape[1] // cube_size[1]
num_cubes_x = data_shape[2] // cube_size[2]

# Create a list of all possible cube indices
cubes = []
particle_cubes = []
non_particle_cubes = []

for z in range(num_cubes_z):
    for y in range(num_cubes_y):
        for x in range(num_cubes_x):
            cubes.append((z, y, x))

# Separate cubes into particle-containing and non-particle cubes
def contains_particle(cube_start, particle_coords):
    for coords in particle_coords.values():
        for coord in coords:
            z, y, x = coord.astype(int)
            if (
                cube_start[0] <= z < cube_start[0] + cube_size[0] and
                cube_start[1] <= y < cube_start[1] + cube_size[1] and
                cube_start[2] <= x < cube_start[2] + cube_size[2]
            ):
                return True
    return False

for cz, cy, cx in cubes:
    cube_start = (cz * cube_size[0], cy * cube_size[1], cx * cube_size[2])
    if contains_particle(cube_start, combined_particle_coords):
        particle_cubes.append((cz, cy, cx))
    else:
        non_particle_cubes.append((cz, cy, cx))

# Limit non-particle cubes to 20% of the dataset
num_non_particle_cubes = int(len(particle_cubes) * 0.1)
selected_non_particle_cubes = random.sample(non_particle_cubes, num_non_particle_cubes)
selected_cubes = particle_cubes + selected_non_particle_cubes
print(f"Selected {len(selected_cubes)} cubes for the dataset. Where {len(particle_cubes)} contain particles and {len(selected_non_particle_cubes)} do not.")


# ------------------- VISUALIZE Combined Tomogram Data ----------------------------#

# Define a color map for label IDs
label_colors = {
    1: "red",        # virus-like-particle
    2: "green",      # apo-ferritin
    3: "blue",       # beta-amylase
    4: "yellow",     # beta-galactosidase
    5: "magenta",    # ribosome
    6: "cyan",       # thyroglobulin
}

# Function to visualize the combined tomogram with particles in 3D using napari
def visualize_combined_tomogram(tomogram_data, particle_coords):
    # Create a napari viewer
    viewer = napari.Viewer()

    # Add the combined tomogram data as a 3D volume
    viewer.add_image(tomogram_data, name="Combined Tomogram")

    # Collect all particle coordinates and their label IDs
    all_particles = []
    all_labels = []
    for particle_type, coords in particle_coords.items():
        label_id = particle_types[particle_type]
        all_particles.extend(coords)
        all_labels.extend([label_id] * len(coords))

    # Convert to numpy arrays
    all_particles = np.array(all_particles)
    all_labels = np.array(all_labels)

    # Assign colors to each particle based on its label ID
    colors = [label_colors[label] for label in all_labels]

    # Add the particles as a 3D points layer with different colors
    if all_particles.size > 0:
        viewer.add_points(
            all_particles,
            name="Particles",
            face_color=colors,
            size=5,
            opacity=0.8,
        )

    # Start the napari event loop
    napari.run()

if VISUALIZE:
    # Visualize the combined tomogram with particles
    print("Visualizing the combined tomogram with particles...")
    visualize_combined_tomogram(combined_tomogram_data, combined_particle_coords)


def visualize_selected_cubes(tomogram_data, particle_coords, selected_cubes, cube_size):
    # Create a napari viewer
    viewer = napari.Viewer()

    # Iterate through the 10 selected cubes and visualize them
    for idx, (cz, cy, cx) in enumerate(selected_cubes[:10]):  # Limit to 10 cubes
        # Define cube boundaries
        z_start, y_start, x_start = cz * cube_size[0], cy * cube_size[1], cx * cube_size[2]
        z_end, y_end, x_end = z_start + cube_size[0], y_start + cube_size[1], x_start + cube_size[2]

        # Extract cube data from the tomogram
        cube_data = tomogram_data[z_start:z_end, y_start:y_end, x_start:x_end]

        # Collect particle coordinates and labels within the cube
        cube_particles = []
        cube_labels = []
        for particle_type, coords in particle_coords.items():
            label_id = particle_types[particle_type]
            for coord in coords:
                z, y, x = coord.astype(int)
                if z_start <= z < z_end and y_start <= y < y_end and x_start <= x < x_end:
                    # Adjust coordinates to cube-local space
                    cube_particles.append([z - z_start, y - y_start, x - x_start])
                    cube_labels.append(label_id)

        # Convert to numpy arrays
        cube_particles = np.array(cube_particles)
        cube_labels = np.array(cube_labels)

        # Assign colors to each particle based on its label ID
        colors = [label_colors[label] for label in cube_labels]

        # Add the cube data as a volume
        viewer.add_image(cube_data, name=f"Cube {idx + 1}", colormap="gray")

        # Add the particles as a points layer
        if cube_particles.size > 0:
            viewer.add_points(
                cube_particles,
                name=f"Particles in Cube {idx + 1}",
                face_color=colors,
                size=5,
                opacity=0.8,
            )

    # Start the napari event loop
    napari.run()

if VISUALIZE:
    # Visualize the first 10 selected cubes
    print("Visualizing 10 selected cubes with particles...")
    visualize_selected_cubes(combined_tomogram_data, combined_particle_coords, selected_cubes, cube_size)

# ---------------------------------------------------------------------------------#

# -------------------Create a PyTorch Dataset for Mini Cubes-----------------------#
class TomogramDatasetMiniCubes(Dataset):
    def __init__(self, tomogram_data, selected_cubes, particle_coords):
        self.tomogram_data = tomogram_data
        self.selected_cubes = selected_cubes
        self.particle_coords = particle_coords
        self.cube_size = (96, 96, 96)
        self.subcube_size = (12, 12, 12)
        self.subcube_volume = np.prod(self.subcube_size)
        self.num_subcubes = (self.cube_size[0] // self.subcube_size[0]) ** 3
        self.background_id = 0
        # Map particle types to integer IDs
        self.particle_type_ids = {ptype: i+1 for i, ptype in enumerate(particle_coords.keys())}

    def __len__(self):
        return len(self.selected_cubes)

    def __getitem__(self, idx):
        cz, cy, cx = self.selected_cubes[idx]
        z_start, z_end = cz * self.cube_size[0], (cz + 1) * self.cube_size[0]
        y_start, y_end = cy * self.cube_size[1], (cy + 1) * self.cube_size[1]
        x_start, x_end = cx * self.cube_size[2], (cx + 1) * self.cube_size[2]

        cube_data = self.tomogram_data[z_start:z_end, y_start:y_end, x_start:x_end]
        cube_start = (z_start, y_start, x_start)
        labels = self.create_mini_cube_labels(cube_start)
        cube_data = np.expand_dims(cube_data, axis=0)  # Add channel dimension

        return (
            torch.tensor(cube_data, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.int64),
        )

    def create_mini_cube_labels(self, cube_start):
        labels = []
        for z in range(0, self.cube_size[0], self.subcube_size[0]):
            for y in range(0, self.cube_size[1], self.subcube_size[1]):
                for x in range(0, self.cube_size[2], self.subcube_size[2]):
                    mini_cube_start = (z, y, x)
                    mini_cube_end = (
                        mini_cube_start[0] + self.subcube_size[0],
                        mini_cube_start[1] + self.subcube_size[1],
                        mini_cube_start[2] + self.subcube_size[2],
                    )
                    label = self.get_label_for_mini_cube(mini_cube_start, mini_cube_end, cube_start)
                    labels.append(label)
        return labels


    def get_label_for_mini_cube(self, mini_cube_start, mini_cube_end, cube_start):
        # Adjust mini_cube_start and mini_cube_end to global coordinates
        mini_cube_start_global = (
            mini_cube_start[0] + cube_start[0],
            mini_cube_start[1] + cube_start[1],
            mini_cube_start[2] + cube_start[2],
        )
        mini_cube_end_global = (
            mini_cube_end[0] + cube_start[0],
            mini_cube_end[1] + cube_start[1],
            mini_cube_end[2] + cube_start[2],
        )

        for particle_type, coords in self.particle_coords.items():
            for coord in coords:
                # Check if the particle falls within the global mini cube bounds
                if (
                    mini_cube_start_global[0] <= coord[0] < mini_cube_end_global[0]
                    and mini_cube_start_global[1] <= coord[1] < mini_cube_end_global[1]
                    and mini_cube_start_global[2] <= coord[2] < mini_cube_end_global[2]
                ):
                    
                    return self.particle_type_ids[particle_type]
        return self.background_id


# Create the updated dataset
particle_dataset_mini_cubes = TomogramDatasetMiniCubes(
    combined_tomogram_data, selected_cubes, combined_particle_coords
)

# Test the dataset
cube_data, labels = particle_dataset_mini_cubes[200]
# print dataset size and shape of
print(f"len(particle_dataset_mini_cubes): {len(particle_dataset_mini_cubes)}")
print("Cube Data Shape:", cube_data.shape)  # Should be (1, 96, 96, 96)
print("Labels Shape:", labels.shape)        # Should be (512,)
# ---------------------------------------------------------------------------------#

# -------------------Defining the Model Architecture-----------------------# 
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

class TransformerSequence(nn.Module):
    def __init__(self, num_layers, decoder_layer):
        super().__init__()
        self.layers = clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt):
        output = tgt

        for mod in self.layers:
            output = mod(output)
        return output

class CnnTokenizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv_1 = nn.Conv3d(1, config.n_embd //  2, kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm3d(config.n_embd // 2)
        self.conv_2 = nn.Conv3d(config.n_embd // 2, config.n_embd // 2, kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm3d(config.n_embd // 2)
        self.conv_3 = nn.Conv3d(config.n_embd // 2, config.n_embd, kernel_size=3, stride=1, padding=1)
        self.bn_3 = nn.BatchNorm3d(config.n_embd)
        self.downsize = nn.Conv3d(config.n_embd, config.n_embd, kernel_size=3, stride=2, padding=1)
        self.slice = nn.Conv3d(config.n_embd, config.n_embd, kernel_size=config.token_width, stride=config.token_width, padding=0) 

    def forward(self, x):
        x = self.conv_1(x)
        # print(f"shape after conv_1: {x.shape}")
        x = self.bn_1(x)
        x = F.gelu(x)
        x = self.conv_2(x)
        # print(f"shape after conv_2: {x.shape}")
        x = self.bn_2(x)
        x = F.gelu(x)
        x = self.conv_3(x)
        # print(f"shape after conv_3: {x.shape}")
        x = self.bn_3(x)
        x = F.gelu(x)
        x = self.downsize(x)
        # print(f"shape after downsize: {x.shape}")
        x = self.slice(x)
        # print(f"shape after slice: {x.shape}")
        return x

@dataclass
class Config:
    block_size: int = 8**3 # max sequence length
    token_width: int = 6 # width of the cube
    n_layer: int = 4 # number of layers
    n_head: int = 16 # number of heads
    n_embd: int = 128 # embedding dimension


class TestModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = CnnTokenizer(config)
        self.positional_embedding = nn.Parameter(torch.zeros(config.block_size, config.n_embd))
        self.transformer = TransformerSequence(config.n_layer, TransformerBlock(config))
        self.decoder = nn.Linear(config.n_embd, len(particle_types) + 1)  # Output layer

    def forward(self, x):
        x = self.tokenizer(x) # (N, 1, 96, 96, 96) -> (N, n_embd, 8, 8, 8)
        x = x.reshape(x.size(0), self.config.n_embd, -1).transpose(1, 2) + self.positional_embedding # (N, n_embd, 8, 8, 8) -> (N, n_embd, 512) -> (N, 512, n_embd)
        x = self.transformer(x) # (N, 512, n_embd) -> (N, 512, n_embd)
        x = self.decoder(x) # (N, 512, n_embd) -> (N, 512, 7)
        return x

# ---------------------------------------------------------------------------------#

# -------------------Training the Model-----------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
config = Config()

# Instantiate the model
model = TestModel(config).to(device)

# Test the model with a sample input
test_input = cube_data.unsqueeze(0).to(device)
print(test_input.shape)
with torch.no_grad():
    output = model(test_input)
print("Model Output Shape:", output.shape)  # Should be (1, 512, 7)

# print the # of parameters
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):.2e}")

# Define optimizer, loss, and scheduler
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
weights = torch.tensor([0.5] + [1 for _ in range(1, len(particle_types) + 1)]).to(device)  # Lower weight for background
print(weights)
criterion = nn.CrossEntropyLoss(weight=weights)

# Define the dataset split and loaders
train_size = int(0.8 * len(particle_dataset_mini_cubes))
print("Train size:", train_size)
val_size = (len(particle_dataset_mini_cubes) - train_size)
print("Validation size:", val_size)
# sys.exit()
train_dataset, val_dataset = random_split(particle_dataset_mini_cubes, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


# Training loop
epochs = 20
best_val_loss = float('inf')
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, len(particle_types) + 1), labels.view(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()


    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, len(particle_types) + 1), labels.view(-1))
            val_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.8f}, Validation Loss: {val_loss / len(val_loader):.8f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Save the trained model
        torch.save(model.state_dict(), "segmentation_model_mini_cubes.pth")
    

# ---------------------------------------------------------------------------------#

# -------------------Visualizing the Model Predictions-----------------------#
def visualize_inference(inputs, labels, predictions, sample_idx):
    """
    Visualize the input data, ground truth, and predictions using napari.
    """
    # Create a napari viewer
    viewer = napari.Viewer()

    # Add input tomogram
    viewer.add_image(inputs.cpu().numpy(), name=f"Sample {sample_idx} - Input", colormap="gray")

    # Add ground truth labels
    viewer.add_labels(labels.cpu().numpy(), name=f"Sample {sample_idx} - Ground Truth")

    # Add predictions
    viewer.add_labels(predictions.cpu().numpy(), name=f"Sample {sample_idx} - Predictions")

    # Start napari viewer
    napari.run()

# Load the trained model
model.load_state_dict(torch.load("segmentation_model_mini_cubes.pth"))
# Select a few samples from training and validation datasets
model.eval()
for i, (inputs, labels) in enumerate(train_loader):
    if i >= 5:  # Visualize only 5 samples
        break

    inputs, labels = inputs[0].to(device), labels[0].to(device)
    # print(inputs.shape, labels.shape)
    labels = labels.reshape(8, 8, 8)  # Reshape to (8, 8, 8) for visualization
    true_labels = torch.zeros_like(inputs[0]).to(torch.int64) # shape (96, 96, 96)
    for z in range(0, 96, 12):
        for y in range(0, 96, 12):
            for x in range(0, 96, 12):
                lz, ly, lx = z//12, y//12, x//12
                true_labels[z:z+12, y:y+12, x:x+12] = labels[lz, ly, lx]
    # visualize_inference(inputs[0], true_labels, true_labels, sample_idx=i)

    inputs = inputs.unsqueeze(1)  # Add channel dimension for model input

    with torch.no_grad():
        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=-1)  # Get predicted class labels
        predictions = predictions.reshape(8, 8, 8)  # Reshape to (8, 8, 8) for visualization
        true_predictions = torch.zeros_like(inputs[0, 0]).to(torch.int64) # shape (96, 96, 96)
        for z in range(0, 96, 12):
            for y in range(0, 96, 12):
                for x in range(0, 96, 12):
                    lz, ly, lx = z//12, y//12, x//12
                    true_predictions[z:z+12, y:y+12, x:x+12] = predictions[lz, ly, lx]

    # Visualize the first sample in the batch
    visualize_inference(inputs[0, 0], true_labels, true_predictions, sample_idx=i)

def visualize_inference(inputs, labels, predictions, sample_idx):
    """
    Visualize the input data, ground truth, and predictions using napari.
    """
    # Create a napari viewer
    viewer = napari.Viewer()

    # Add input tomogram
    viewer.add_image(inputs.cpu().numpy(), name=f"Sample {sample_idx} - Input", colormap="gray")

    # Add ground truth labels
    viewer.add_labels(labels.cpu().numpy(), name=f"Sample {sample_idx} - Ground Truth")

    # Add predictions
    viewer.add_labels(predictions.cpu().numpy(), name=f"Sample {sample_idx} - Predictions")

    # Start napari viewer
    napari.run()

# Load the trained model
model.load_state_dict(torch.load("segmentation_model_mini_cubes.pth"))
# Select a few samples from training and validation datasets
model.eval()
for i, (inputs, labels) in enumerate(val_loader):
    if i >= 5:  # Visualize only 5 samples
        break

    inputs, labels = inputs[0].to(device), labels[0].to(device)
    labels = labels.reshape(8, 8, 8)  # Reshape to (8, 8, 8) for visualization
    true_labels = torch.zeros_like(inputs[0]).to(torch.int64) # shape (96, 96, 96)
    for z in range(0, 96, 12):
        for y in range(0, 96, 12):
            for x in range(0, 96, 12):
                lz, ly, lx = z//12, y//12, x//12
                true_labels[z:z+12, y:y+12, x:x+12] = labels[lz, ly, lx]
 

    inputs = inputs.unsqueeze(1)  # Add channel dimension for model input

    with torch.no_grad():
        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=-1)  # Get predicted class labels
        predictions = predictions.reshape(8, 8, 8)  # Reshape to (8, 8, 8) for visualization
        true_predictions = torch.zeros_like(inputs[0, 0]).to(torch.int64) # shape (96, 96, 96)
        for z in range(0, 96, 12):
            for y in range(0, 96, 12):
                for x in range(0, 96, 12):
                    lz, ly, lx = z//12, y//12, x//12
                    true_predictions[z:z+12, y:y+12, x:x+12] = predictions[lz, ly, lx]

    # Visualize the first sample in the batch
    visualize_inference(inputs[0, 0], true_labels, true_predictions, sample_idx=i)

# ---------------------------------------------------------------------------------#