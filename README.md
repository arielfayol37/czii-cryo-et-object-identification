# Cryo-ET Object Identification Project (kaggle link: https://www.kaggle.com/c/czii-cryo-et-object-identification/data)

## Overview

This project implements a deep learning solution for identifying particle centers in 3D cryo-electron tomography (cryo-ET) data. The goal is to detect and classify 5 different particle types in tomographic reconstructions, with a focus on achieving high precision and recall for particle localization.

# Quick Takeaway
I heard about this competition 2 weeks before the deadline and didn't have enough time to make the model work well but I learned the following: the transformer is incredibly good at memorization. I posit that the argument "transformers need a lot of data to learn" is just a by-product/symptom of their memorization ability. If I were to work further on this project,
I would approach training differently: instead of having a fixed dataset, I will just be applying random augmentations like rotations, clipping, adding noise, during traning and not before.

## Dataset Description
(Note: Go on the Kaggle website to obtain the dataset)
The competition focuses on identifying points corresponding to particle centers in 3D tomograms. The dataset contains 6 particle types with different difficulty levels:

- **apo-ferritin** (easy)
- **beta-amylase** (impossible, not scored)
- **beta-galactosidase** (hard)
- **ribosome** (easy)
- **thyroglobulin** (hard)
- **virus-like-particle** (easy)

**Note**: beta-amylase is not scored as it's considered too difficult for accurate evaluation.

### Data Structure

```
train/
├── static/ExperimentRuns/{experiment}/VoxelSpacing10.000/
│   └── denoised.zarr/ - zarr files containing tomograph data
└── overlay/ExperimentRuns/{experiment}/Picks/
    └── {particle_type}.json - ground truth files with particle coordinates

test/
├── static/ExperimentRuns/{experiment}/VoxelSpacing10.000/
│   └── denoised.zarr/ - zarr files containing tomograph data
└── sample_submission.csv - submission file format
```

### Tomograms
- 3D arrays provided as multiscale 3D OME-NGFF Zarr arrays
- Can be opened with zarr, ome-zarr, and copick libraries
- Training data includes 4 filtered versions: denoised, wbp, ctfdeconvolved, isonetcorrected
- Test data only includes denoised tomograms

### Particle Locations
- Stored in JSON files according to copick specification
- Can be converted to segmentation masks using copick utilities

## Project Architecture

### Model Architecture

The project implements a transformer-based architecture for 3D segmentation:

#### Core Components

1. **LinearTokenizer** (`model.py`)
   - Converts 3D input cubes (96×96×96) into token sequences
   - Uses 3D convolutions with configurable token width (default: 6)
   - Outputs sequence of tokens for transformer processing

2. **Transformer Architecture**
   - **SelfAttention**: Multi-head attention mechanism with flash attention
   - **MLP**: Feed-forward network with GELU activation
   - **TransformerBlock**: Combines attention and MLP with residual connections
   - **Positional Embeddings**: Learnable positional encodings

3. **Model Variants**
   - **BaseModel**: Standard transformer for segmentation
   - **ContrastiveModel**: Adds summarizer token and projection head for contrastive learning
   - **SegmentationModel**: Inherits from ContrastiveModel, supports both summarizer and non-summarizer modes

#### Configuration

```python
@dataclass
class LinearConfig:
    block_size: int = int((96**3)/(6**3))  # max sequence length (4096)
    token_width: int = 6                    # width of the cube
    n_layer: int = 6                        # number of transformer layers
    n_head: int = 4                         # number of attention heads
    n_embd: int = 128                       # embedding dimension
    n_class: int = 7                        # number of classes (6 particles + background)
```

### Data Processing Pipeline

#### 1. Data Generation (`generate_training_data.py`, `generate_pretraining_data.py`)

**Training Data Generation:**
- Loads tomogram data from Zarr files
- Processes particle coordinates from JSON files
- Creates 96×96×96 cubes with 6×6×6 subcubes for labeling
- Generates segmentation labels based on particle presence
- Saves processed data as PyTorch tensors

**Pretraining Data Generation:**
- Uses segmentation masks instead of point coordinates
- Processes larger dataset (27 experiments vs 7 for training)
- Creates similar cube-based dataset for pretraining

#### 2. Data Augmentation (`training.py`)

Implements comprehensive 3D data augmentation:
- **Random Rotation**: 3D rotation along random axes (±30°)
- **Random Flip**: Flips along any of the 3 axes
- **Random Translation**: Shifts data with edge handling
- **Gaussian Noise**: Adds noise to input data

#### 3. Dataset Classes

- **TomogramDatasetMiniCubes**: Base dataset for cube-based processing
- **CustomTransformDataset**: Applies augmentations during training
- **AugmentedTomogramDataset**: Alternative implementation using torchio

### Training Pipeline

#### 1. Main Training (`training.py`)

**Key Features:**
- Uses CrossEntropyLoss with class weights for imbalanced data
- Implements F-beta score calculation (β=4) for evaluation
- Supports both CrossEntropy and Tversky loss functions
- Learning rate scheduling (halves at 50% of training)
- Gradient clipping for stability
- Model checkpointing based on validation loss

**Training Configuration:**
- Batch size: 16
- Learning rate: 6e-5
- Weight decay: 0.01
- Epochs: 10,000
- Mixed precision training (bfloat16)

#### 2. Pretraining (`pretraining.py`)

**Key Features:**
- Uses Tversky loss for better handling of class imbalance
- Larger batch size (64) for pretraining
- Higher learning rate (1e-3) initially
- Torch compilation for optimization

#### 3. Alternative Training (`transformer_train.py`)

**Key Features:**
- CNN-based tokenizer instead of linear tokenizer
- Different model architecture with more attention heads
- Smaller batch size (4) for memory constraints
- Visualization capabilities with napari

### Loss Functions and Metrics

#### 1. Loss Functions

- **CrossEntropyLoss**: Standard classification loss with class weights
- **Weighted Tversky Loss**: Custom loss for handling class imbalance
  - Parameters: α=0.8 (false negative penalty), β=0.2 (false positive penalty)
  - Supports class-specific weights

#### 2. Evaluation Metrics

- **F-beta Score**: β=4, emphasizing recall over precision
- **Per-class Recall**: Individual recall for each particle type
- **Class Weights**: Inverse frequency weighting for imbalanced classes

#### 3. Class Weighting Strategy

```python
# F-beta weights (0 for background, higher for hard classes)
fbeta_weights = torch.tensor([0, 1, 1, 0, 2, 1, 2])

# Training weights (combination of inverse frequency and importance)
weights = torch.tensor([1, 1, 1, 1, 2, 1, 2]) * inv_freq_class_weights
```

### Utilities (`utils.py`)

#### Key Functions

1. **compute_fbeta_loss**: Calculates F-beta score with β=4
2. **compute_recall_per_class**: Computes per-class recall metrics
3. **weighted_tversky_loss**: Custom loss function for segmentation
4. **calculate_class_weights**: Computes inverse frequency weights
5. **plot_label_distribution_torch**: Visualizes class distribution

## File Structure

```
├── model.py                    # Core model architecture
├── utils.py                    # Utility functions and metrics
├── training.py                 # Main training script
├── pretraining.py              # Pretraining script
├── transformer_train.py        # Alternative training with CNN tokenizer
├── generate_training_data.py   # Training data generation
├── generate_pretraining_data.py # Pretraining data generation
├── test_gpt.py                 # Testing script with torchio augmentation
├── development.ipynb           # Main development notebook
├── development_1.ipynb         # Additional development notebook
├── pretraining.ipynb           # Pretraining notebook
├── playground.ipynb            # Experimental notebook
├── sample_submission.csv       # Submission format example
├── label_distribution.png      # Class distribution visualization
└── .gitignore                  # Git ignore rules
```

## Usage

### Prerequisites

```bash
pip install torch torchvision torchaudio
pip install zarr ome-zarr copick
pip install napari rich matplotlib scipy
pip install torchio monai
```

### Data Preparation

1. **Generate Training Data:**
   ```bash
   python generate_training_data.py
   ```

2. **Generate Pretraining Data (optional):**
   ```bash
   python generate_pretraining_data.py
   ```

### Training

1. **Main Training:**
   ```bash
   python training.py
   ```

2. **Pretraining:**
   ```bash
   python pretraining.py
   ```

3. **Alternative Training:**
   ```bash
   python transformer_train.py
   ```

### Model Files

The training scripts save model checkpoints:
- `segmentation_model_mini_cubes.pth`: Best model based on validation loss
- `direct_model_mini_cubes.pth`: Final model after training

## Key Features

### 1. Multi-Scale Processing
- Processes 96×96×96 input cubes
- Uses 6×6×6 subcubes for fine-grained labeling
- Supports different tokenization strategies

### 2. Robust Data Augmentation
- 3D rotations, flips, and translations
- Gaussian noise injection
- Maintains spatial consistency between input and labels

### 3. Class Imbalance Handling
- Inverse frequency weighting
- Custom Tversky loss
- F-beta evaluation (β=4) emphasizing recall

### 4. Visualization
- 3D visualization using napari
- Label distribution plots
- Training progress monitoring

### 5. Flexible Architecture
- Modular transformer design
- Configurable hyperparameters
- Support for different tokenization methods

## Performance Considerations

### Memory Optimization
- Mixed precision training (bfloat16)
- Gradient clipping
- Configurable batch sizes
- Efficient data loading

### Training Stability
- Learning rate scheduling
- Weight decay regularization
- Early stopping based on validation loss
- Reproducible random seeds

## Future Improvements

1. **Model Architecture**
   - Experiment with different tokenization strategies
   - Try attention mechanisms optimized for 3D data
   - Implement hierarchical processing

2. **Data Processing**
   - Multi-scale training
   - Advanced augmentation techniques
   - Semi-supervised learning approaches

3. **Evaluation**
   - Implement additional metrics
   - Cross-validation strategies
   - Ensemble methods

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


## Acknowledgments

- Dataset provided by the cryo-ET competition organizers
- Built with PyTorch, napari, and other open-source libraries
- Inspired by transformer architectures for computer vision tasks
