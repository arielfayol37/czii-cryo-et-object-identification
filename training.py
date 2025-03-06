# Library imports
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

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
import napari
from utils import *
from scipy.ndimage import rotate
from model import SegmentationModel, LinearConfig

# set torch and cuda seed for reproducibility
torch.manual_seed(37)
torch.cuda.manual_seed(37)
random.seed(37)
np.random.seed(37)

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



def random_rotation_3d(image, label, max_angle=30):
    """
    Randomly rotate both the image and label by the same angle along a random axis.

    Parameters:
        image (np.ndarray): The 3D input image of shape (D, H, W).
        label (np.ndarray): The 3D label mask of shape (D, H, W).
        max_angle (float): The maximum rotation angle in degrees.
    
    Returns:
        Rotated image and label.
    """
    # Choose a random axis and corresponding plane for rotation
    axis = random.choice([0, 1, 2])  # Randomly select the rotation axis
    if axis == 0:
        axes = (1, 2)  # Rotate in the y-z plane
    elif axis == 1:
        axes = (0, 2)  # Rotate in the x-z plane
    else:
        axes = (0, 1)  # Rotate in the x-y plane

    # Random angle within the specified range
    angle = random.uniform(-max_angle, max_angle)

    # Rotate the image and label using scipy.ndimage.rotate
    rotated_image = rotate(image, angle, axes=axes, reshape=False, mode='constant', order=1)
    rotated_label = rotate(label, angle, axes=axes, reshape=False, mode='constant', order=0)  # No interpolation for labels
    
    return rotated_image, rotated_label

def random_flip_3d(image, label):
    """
    Randomly flip both the image and label along each axis with 50% probability.
    
    Parameters:
        image (np.ndarray): The 3D input image of shape (D, H, W).
        label (np.ndarray): The 3D label mask of shape (D, H, W).
    
    Returns:
        Flipped image and label.
    """
    # Randomly flip along each axis
    for axis in range(3):
        if random.random() > 0.5:
            image = np.flip(image, axis=axis)
            label = np.flip(label, axis=axis)
    return image, label


def random_translation_3d(image, label, max_shift=10, subcube_size=6):
    """
    Randomly translate both the image and label by the same amount along each axis.

    Parameters:
        image (np.ndarray): The 3D input image of shape (96, 96, 96).
        label (np.ndarray): The 3D label cube of shape (16, 16, 16).
        max_shift (int): Maximum number of voxels to shift along any axis.
        subcube_size (int): Size of the subcubes in the input corresponding to each label (typically 6).
    
    Returns:
        Translated image and label.
    """
    # Random shifts along each axis
    voxel_shifts = [random.randint(-max_shift, max_shift) for _ in range(3)]
    label_shifts = [shift // subcube_size for shift in voxel_shifts]  # Scale down shifts for labels

    # Apply the same translation to both image and label
    translated_image = np.roll(image, shift=voxel_shifts, axis=(0, 1, 2))
    translated_label = np.roll(label, shift=label_shifts, axis=(0, 1, 2))

    # Fill edge regions with background class (0) to handle wrapping
    for dim, shift in enumerate(voxel_shifts):
        if shift > 0:
            translated_image[:shift, :, :] = 0  # Zero out the starting region
            translated_label[:label_shifts[dim], :, :] = 0  # Zero out the label region
        elif shift < 0:
            translated_image[shift:, :, :] = 0  # Zero out the ending region
            translated_label[label_shifts[dim]:, :, :] = 0  # Zero out the label region

    return translated_image, translated_label


def add_gaussian_noise(image, mean=0.0, std=0.1):
    """
    Add Gaussian noise to the image.

    Parameters:
        image (np.ndarray): The 3D input image of shape (D, H, W).
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.
    
    Returns:
        Noisy image.
    """
    noise = np.random.normal(mean, std, image.shape)
    return image + noise


class CustomTransformDataset(Dataset):
    def __init__(self, tomogram_data, segmentation_labels, transform=None):
        assert tomogram_data.size(0) == segmentation_labels.size(0), f"{tomogram_data.size(0)}, {segmentation_labels.size(0)}"
        self.tomogram_data = tomogram_data
        self.segmentation_labels = segmentation_labels
        self.transform = transform

    def __len__(self):
        return self.tomogram_data.size(0)

    def __getitem__(self, idx):
        cube, labels = self.tomogram_data[idx].squeeze().numpy(), self.segmentation_labels[idx].reshape((16, 16, 16)).numpy()

        # Apply augmentations only if transform is specified
        if self.transform:
            cube, labels = self.transform(cube, labels)
        # Return the transformed tensors
        return torch.tensor(cube, dtype=torch.float32).unsqueeze(0), torch.tensor(labels.reshape(-1), dtype=torch.int64)
    
# Define the training data augmentation
def training_transform(image, label):
    # Apply custom transformations (rotation, flip, translation, noise)
    image, label = random_rotation_3d(image, label, max_angle=30)
    image, label = random_flip_3d(image, label)
    image, label = random_translation_3d(image, label, max_shift=6)
    image = add_gaussian_noise(image, mean=0.0, std=0.1)  # Noise only to the image
    return image, label

# No transformations for validation
def validation_transform(image, label):
    return image, label  # Just return the original image and label without augmentation

input_data, segmentation_labels = torch.load("segmentation_training_input_data.pt"), torch.load("segmentation_training_labels.pt")
# Properly split the data
train_size = int(0.8 * len(input_data))
val_size = len(input_data) - train_size
train_data, val_data = random_split(list(zip(input_data, segmentation_labels)), [train_size, val_size])

# Create separate datasets
train_input, train_labels = zip(*train_data)
val_input, val_labels = zip(*val_data)

train_dataset = CustomTransformDataset(torch.stack(train_input), torch.stack(train_labels), transform=training_transform)
val_dataset = CustomTransformDataset(torch.stack(val_input), torch.stack(val_labels), transform=validation_transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"Train Size: {len(train_dataset)}, Val Size: {len(val_dataset)}")

# Example: Check a training sample and a validation sample
train_sample_input, train_sample_label = train_dataset[0]
val_sample_input, val_sample_label = val_dataset[0]

print("Training Sample Shape:", train_sample_input.shape, train_sample_label.shape)
print("Validation Sample Shape:", val_sample_input.shape, val_sample_label.shape)
# sys.exit()
particle_dataset_mini_cubes = CustomTransformDataset(input_data, segmentation_labels)
# sample_input, sample_label = particle_dataset_mini_cubes[0]
# print(sample_input.shape, sample_label.shape)
# sys.exit()

labels_tensor = get_all_labels(particle_dataset_mini_cubes)
plot_label_distribution_torch(labels_tensor)
inv_freq_class_weights = calculate_class_weights(labels_tensor, 7, device)
# print(f"Inv Freq Weights: {inv_freq_class_weights}")
# sys.exit()

torch.set_float32_matmul_precision("high")


config = LinearConfig()
segmentation_model = SegmentationModel(config).to(device)
# segmentation_model = torch.compile(segmentation_model)
learning_rate = 6e-5


# Define optimizer, loss, and scheduler
optimizer = segmentation_model.configure_optimizers(weight_decay=0.01, learning_rate=learning_rate, device=device)
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
"""
particle_types = {"virus-like-particle": 1, "apo-ferritin": 2, "beta-amylase": 3, 
                  "beta-galactosidase": 4, "ribosome": 5, "thyroglobulin": 6}
"""
# background, virus-like particle, "apo-ferritin", "beta-amylase","beta-galactosidase","ribosome","thyroglobulin"
fbeta_weights = torch.tensor([0, 1, 1, 0, 2, 1, 2]).to(device)
print(f"beta weights: {fbeta_weights}")

weights = torch.tensor([1, 1, 1, 1, 2, 1, 2]).to(device) * inv_freq_class_weights
print(f"Weights: {weights}")
# weights = torch.tensor([0.001, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# Training loop
epochs = 10000
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
            loss = criterion(outputs, labels)
            # loss = weighted_tversky_loss(pred = outputs, target = labels, class_weights = weights)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(segmentation_model.parameters(), max_norm=1.0)
        # print("norm: ", norm)
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

torch.save(segmentation_model.state_dict(), "direct_model_mini_cubes.pth")
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

print(segmentation_model)
# Load the trained model
segmentation_model.load_state_dict(torch.load("direct_model_mini_cubes.pth"))
# Select a few samples from training and validation datasets
segmentation_model.eval()
for i, (inputs, labels) in enumerate(train_loader):
    if i >= 3:  # Visualize only 10 samples
        break

    inputs, labels = inputs[0].to(device), labels[0].to(device)
    # print(inputs.shape, labels.shape)
    labels = labels.reshape(16, 16, 16)  # Reshape to (16, 16, 16) for visualization
    true_labels = torch.zeros_like(inputs[0]).to(torch.int64) # shape (96, 96, 96)
    # Iterate over the (16, 16, 16) labels and assign them to (6, 6, 6) regions
    for z in range(16):
        for y in range(16):
            for x in range(16):
                true_labels[z*6:(z+1)*6, y*6:(y+1)*6, x*6:(x+1)*6] = labels[z, y, x]
    # visualize_inference(inputs[0], true_labels, true_labels, sample_idx=i)

    inputs = inputs.unsqueeze(1)  # Add channel dimension for model input

    with torch.no_grad():
        outputs = segmentation_model(inputs)
        predictions = torch.argmax(outputs, dim=-1)  # Get predicted class labels
        predictions = predictions.reshape(16, 16, 16)  # Reshape to (16, 16, 16) for visualization
        true_predictions = torch.zeros_like(inputs[0, 0]).to(torch.int64) # shape (96, 96, 96)
        for z in range(16):
            for y in range(16):
                for x in range(16):
                    true_predictions[z*6:(z+1)*6, y*6:(y+1)*6, x*6:(x+1)*6] = predictions[z, y, x]

    # Visualize the first sample in the batch
    visualize_inference(inputs[0, 0], true_labels, true_predictions, sample_idx=i)