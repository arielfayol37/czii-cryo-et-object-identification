# Library imports
import random
import torch
import numpy as np
import zarr
import json
import os
import napari 
# set torch and cuda seed for reproducibility
torch.manual_seed(37)
torch.cuda.manual_seed(37)
COUNTER = 0

# -------------LOADING TOMOGRAM DATA AND PARTICLE COORDINATES-----------------#

# Define the experiment runs to load
experiment_runs = ["TS_5_4", "TS_69_2", "TS_6_4", "TS_6_6", "TS_73_6", "TS_86_3", "TS_99_9"]
particle_types = {"virus-like-particle": 1, "apo-ferritin": 2, "beta-amylase": 3, "beta-galactosidase": 4, "ribosome": 5, "thyroglobulin": 6}
particle_radii = {"virus-like-particle": 135, "apo-ferritin": 60, "beta-amylase": 65, "beta-galactosidase": 90, "ribosome": 150, "thyroglobulin": 130}
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
        tomogram_data = (tomogram_data - tomogram_data.min()) / (tomogram_data.max() - tomogram_data.min())
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
            coords = np.round([
                [
                    (p["location"]["z"] / voxel_spacing[0]) + cumulative_z_depth,
                    p["location"]["y"] / voxel_spacing[1],
                    p["location"]["x"] / voxel_spacing[2],
                ]
                for p in points
            ]).astype(int)

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


# -------------PRECOMPUTE LABEL CUBE-----------------#
label_cube = np.zeros(combined_tomogram_data.shape, dtype=int)

for particle_type, coords in combined_particle_coords.items():
    particle_id = particle_types[particle_type]
    radius = int(particle_radii[particle_type] / voxel_spacing[0])  # Convert radius to voxel units
    radius = 3 # TODO: change this if you care more than the center
    for coord in coords:
        z, y, x = coord.astype(int)

        # Define the bounding box for the particle
        z_min, z_max = max(0, z - radius), min(label_cube.shape[0], z + radius + 1)
        y_min, y_max = max(0, y - radius), min(label_cube.shape[1], y + radius + 1)
        x_min, x_max = max(0, x - radius), min(label_cube.shape[2], x + radius + 1)
        # print(z_min, z_max, y_min, y_max, x_min, x_max, particle_id)
        # Mark the region with the particle ID
        label_cube[z_min:z_max, y_min:y_max, x_min:x_max] = particle_id

print("Label cube precomputed.")


# -------------Combine tomograms and sample cubes with particles in it-----------------#

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
num_non_particle_cubes = int(len(particle_cubes) * 0.1)
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
        global COUNTER
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
                        if label != self.background_id and coverage / total_voxels >= 0.3 and coverage > max_coverage:
                            dominant_label = label
                            max_coverage = coverage
                    if dominant_label !=0 : COUNTER += 1
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
torch.save(input_data, "segmentation_training_input_data.pt")
torch.save(segmentation_labels, "segmentation_training_labels.pt")
print(COUNTER)
