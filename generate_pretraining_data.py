# Library imports
import random
import torch
import numpy as np
import zarr
import os
# -------------LOADING TOMOGRAM DATA AND SEGMENTATION MASKS-----------------#

experiment_runs = ["TS_" + str(i) for i in range(27)]
particle_types = {"virus-like-particle": 1, "apo-ferritin": 2, "beta-amylase": 3, 
                  "beta-galactosidase": 4, "ribosome": 5, "thyroglobulin": 6}
type_mapping = {
    "virus-like-particle": ["pp7_vlp", 6], "apo-ferritin": ["ferritin_complex", 1], 
    "beta-amylase": ["beta_amylase", 2], "beta-galactosidase": ["beta_galactosidase", 3], 
    "ribosome": ["cytosolic_ribosome", 4], "thyroglobulin": ["thyroglobulin", 5]
}

# Initialize lists to store tomograms and labels
combined_tomogram_data = []
combined_label_data = []

for experiment_run in experiment_runs:
    zarr_file_path = os.path.join("aws_data", "10441", experiment_run, "Reconstructions", 
                                  "VoxelSpacing10.000", "Tomograms", "100", experiment_run)

    try:
        # Load tomogram
        tomogram = zarr.open(zarr_file_path + ".zarr", mode="r")
        tomogram_data = tomogram["0"][:]
        # tomogram_data = (tomogram_data - tomogram_data.mean()) / tomogram_data.std()
        
        # Initialize label cube for this tomogram
        label_cube = np.zeros_like(tomogram_data, dtype=np.uint8)

        # Process segmentation masks
        for particle_type, particle_id in particle_types.items():
            name, number = type_mapping[particle_type]
            segmentation_mask_path = os.path.join("aws_data", "10441", experiment_run, "Reconstructions", 
                                                  "VoxelSpacing10.000", "Annotations", 
                                                  "10" + str(number), name + "-1.0_segmentationmask.zarr")
            try:
                segmentation_mask = zarr.open(segmentation_mask_path, mode="r")["0"][:]
                label_cube[segmentation_mask > 0] = particle_id  # Assign particle ID where mask exists
            except Exception as e:
                print(f"Error loading segmentation mask for {particle_type} in {experiment_run}: {e}")

        # Append processed data
        combined_tomogram_data.append(tomogram_data)
        combined_label_data.append(label_cube)

    except Exception as e:
        print(f"Error loading Zarr file for {experiment_run}: {e}")

# Concatenate all tomograms and labels along the Z-dimension
combined_tomogram_data = np.concatenate(combined_tomogram_data, axis=0)
combined_label_data = np.concatenate(combined_label_data, axis=0)

print("Final combined tomogram shape:", combined_tomogram_data.shape)
print("Final combined label shape:", combined_label_data.shape)


# -------------Combine tomograms and sample cubes with particles in it-----------------#
label_cube = combined_label_data
# Dimensions of the combined tomogram data
data_shape = combined_tomogram_data.shape
cube_size = (96, 96, 96)
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
def contains_particle(cube_start, label_cube):
    z_start, y_start, x_start = cube_start
    z_end, y_end, x_end = z_start + cube_size[0], y_start + cube_size[1], x_start + cube_size[2]
    return np.any(label_cube[z_start:z_end, y_start:y_end, x_start:x_end] > 0)

for cz, cy, cx in cubes:
    cube_start = (cz * cube_size[0], cy * cube_size[1], cx * cube_size[2])
    if contains_particle(cube_start, label_cube):
        particle_cubes.append((cz, cy, cx))
    else:
        non_particle_cubes.append((cz, cy, cx))

# Limit non-particle cubes to 10% of the dataset
num_non_particle_cubes = min(len(non_particle_cubes), int(len(particle_cubes) * 0.1))

selected_non_particle_cubes = random.sample(non_particle_cubes, num_non_particle_cubes)
selected_cubes = particle_cubes + selected_non_particle_cubes
print(f"Selected {len(selected_cubes)} cubes for the dataset. Where {len(particle_cubes)} contain particles and {len(selected_non_particle_cubes)} do not.")


class DataCreator():
    def __init__(self, tomogram_data, label_cube, selected_cubes):
        self.tomogram_data = tomogram_data
        self.label_cube = label_cube
        self.selected_cubes = selected_cubes
        self.cube_size = (96, 96, 96)
        self.subcube_size = (6, 6, 6)
        self.background_id = 0

    def __len__(self):
        return len(self.selected_cubes)

    def getitem(self, idx):
        cz, cy, cx = self.selected_cubes[idx]
        z_start, z_end = cz * self.cube_size[0], (cz + 1) * self.cube_size[0]
        y_start, y_end = cy * self.cube_size[1], (cy + 1) * self.cube_size[1]
        x_start, x_end = cx * self.cube_size[2], (cx + 1) * self.cube_size[2]

        cube_data = self.tomogram_data[z_start:z_end, y_start:y_end, x_start:x_end]
        labels = self.generate_labels(z_start, y_start, x_start)

        cube_data = np.expand_dims(cube_data, axis=0)  # Add channel dimension
        return (
            torch.tensor(cube_data, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.int64),
        )

    def generate_labels(self, z_start, y_start, x_start):
        mini_cube_labels = []
        
        for z in range(0, self.cube_size[2], self.subcube_size[2]):  # Width first
            for y in range(0, self.cube_size[1], self.subcube_size[1]):  # Height second
                for x in range(0, self.cube_size[0], self.subcube_size[0]):  # Depth last
                    mini_cube = self.label_cube[
                        z_start + z:z_start + z + self.subcube_size[0],
                        y_start + y:y_start + y + self.subcube_size[1],
                        x_start + x:x_start + x + self.subcube_size[2],
                    ]
                    unique, counts = np.unique(mini_cube, return_counts=True)
                    label_coverage = dict(zip(unique, counts))
                    total_voxels = np.prod(self.subcube_size)

                    dominant_label = self.background_id
                    max_coverage = 0

                    for label, coverage in label_coverage.items():
                        if label != self.background_id and coverage / total_voxels >= 0.37 and coverage > max_coverage:
                            dominant_label = label
                            max_coverage = coverage

                    mini_cube_labels.append(dominant_label)

        return np.array(mini_cube_labels)

    def generate_data(self):
        tomogram_data = torch.zeros((len(self.selected_cubes), 1, *self.cube_size))
        segmentation_labels = torch.zeros((len(self.selected_cubes), int(self.cube_size[0]**3/self.subcube_size[0]**3)), dtype=torch.int64)

        for idx in range(len(self.selected_cubes)):
            data_tensor, label_tensor= self.getitem(idx)
            tomogram_data[idx] = data_tensor
            segmentation_labels[idx] = label_tensor
        return tomogram_data, segmentation_labels

# Data Creator
data_creator = DataCreator(
    combined_tomogram_data, label_cube, selected_cubes
)

input_data, segmentation_labels = data_creator.generate_data()

print(input_data.shape, segmentation_labels.shape)
torch.save(input_data, "segmentation_input_data.pt")
torch.save(segmentation_labels, "segmentation_labels.pt")
del combined_tomogram_data
del label_cube
del combined_label_data