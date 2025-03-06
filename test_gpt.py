import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchio as tio  # For data augmentation
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(37)
torch.cuda.manual_seed(37)

# -------------------- DATA AUGMENTATION ------------------- #
augmentation = tio.Compose([
    tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),  # Random flips along all axes
    tio.RandomAffine(scales=(0.9, 1.1), degrees=10),  # Random scaling and rotations
    tio.RandomNoise(std=0.05),  # Random Gaussian noise
])

# -------------------- DATASET CLASS ------------------- #
class AugmentedTomogramDataset(Dataset):
    def __init__(self, tomogram_data, segmentation_labels, augment=False):
        self.tomogram_data = tomogram_data
        self.segmentation_labels = segmentation_labels
        self.augment = augment
        print(tomogram_data.shape, segmentation_labels.shape)

    def __len__(self):
        return self.tomogram_data.size(0)

    def __getitem__(self, idx):
        data, label = self.tomogram_data[idx], self.segmentation_labels[idx].reshape(16, 16, 16)

        # Unsqueeze to add channel dimension (from [16, 16, 16] to [1, 16, 16, 16])
        data = data.unsqueeze(0)  # For image: [1, 96, 96, 96]
        label = label.unsqueeze(0)  # For label: [1, 16, 16, 16]

        if self.augment:
            # Wrap in torchio.Subject with images and labels
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=data),
                label=tio.LabelMap(tensor=label)
            )

            # Apply augmentation
            augmented_subject = augmentation(subject)
            data = augmented_subject.image.tensor  # Extract augmented image
            label = augmented_subject.label.tensor  # Extract augmented label

        # Reshape the labels from [1, 16, 16, 16] to [4096, 1]
        label = label.view(-1, 1)
        return data, label

# -------------------- TESTING THE FIX ------------------- #
# Dummy example to verify the fix
dummy_tomogram = torch.randn(10, 1, 96, 96, 96)  # Single-channel 3D data
dummy_labels = torch.randint(0, 2, (10, 16, 16, 16))  # Corresponding label

dataset = AugmentedTomogramDataset(dummy_tomogram, dummy_labels, augment=True)
data, label = dataset[0]
print("Data shape:", data.shape)  # Should be [1, 96, 96, 96]
print("Label shape:", label.shape)  # Should be [4096, 1]
