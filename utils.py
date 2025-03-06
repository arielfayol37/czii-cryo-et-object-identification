import torch
import matplotlib.pyplot as plt        
def compute_fbeta_loss(predictions: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor, num_classes: int, device):
    """
    Computes F-beta score where beta = 4
    """
    beta = 4
    
    # Initialize totals for true positives, false positives, false negatives, and actual positives
    true_p_total = torch.zeros(1, device=device)
    false_p_total = torch.zeros(1, device=device)
    false_n_total = torch.zeros(1, device=device)
    actual_p_total = torch.zeros(1, device=device)
    
    for class_id in range(num_classes):
        # True positives: predicted class matches ground truth
        true_positives = torch.sum((labels == class_id) & (predictions == class_id)).float()
        # False positives: predicted class matches, but ground truth is not this class
        false_positives = torch.sum((labels != class_id) & (predictions == class_id)).float()
        # False negatives: ground truth is this class, but predicted class is not
        false_negatives = torch.sum((labels == class_id) & (predictions != class_id)).float()
        # Actual positives: ground truth is this class
        actual_positives = torch.sum(labels == class_id).float()

        # Weighted sums
        true_p_total += weights[class_id] * true_positives
        false_p_total += weights[class_id] * false_positives
        false_n_total += weights[class_id] * false_negatives
        actual_p_total += weights[class_id] * actual_positives
    
    # Calculate precision and recall
    precision = true_p_total / (true_p_total + false_p_total + 1e-8)  # Adding small epsilon to avoid division by zero
    recall = true_p_total / (true_p_total + false_n_total + 1e-8)  # Adding small epsilon to avoid division by zero

    # F-beta score calculation
    f_beta_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)  # Adding small epsilon to avoid division by zero
    return f_beta_score
    
    
def compute_recall_per_class(predictions: torch.Tensor, labels: torch.Tensor, num_classes: int):
    """
    Compute the recall accuracy for each class using tensor operations and return a dictionary.

    Args:
        labels (torch.Tensor): A 1D torch tensor of true labels of shape (N,).
        predictions (torch.Tensor): A 1D torch tensor of predicted labels of shape (N,).
        num_classes (int): Total number of classes.

    Returns:
        dict: A dictionary containing recall for each class, e.g., {0: 0.8, 1: 0.4, ..., num_classes-1: recall}.
    """
    mapping = {0:"background", 1:"virus-like-particle", 2:"apo-ferritin", 3:"beta-amylase", 4:"beta-galactosidase",
               5:"ribosome", 6:"thyroglobulin"}
    # Flatten tensors to ensure they are 1D
    labels = labels.view(-1)
    predictions = predictions.view(-1)
    
    recall_per_class = {}
    # Vectorized computation for true positives and actual positives
    for class_id in range(num_classes):
        true_positives = torch.sum((labels == class_id) & (predictions == class_id)).item()
        actual_positives = torch.sum(labels == class_id).item()
        
        if actual_positives != 0:
            recall_per_class[class_id] = true_positives/actual_positives

    recall_dict = {}
    # Convert to a dictionary with class IDs as keys and recall values as values
    for class_id in range(num_classes):
        recall_dict[mapping[class_id]] = f"{recall_per_class[class_id]:.6f}" if class_id in recall_per_class else "7.777777"
        
    return recall_dict

def get_all_labels(dataset):
    """
    Concatenates all labels from the dataset into a single tensor.

    Parameters:
        dataset (torch.utils.data.Dataset): The dataset containing input data and labels.

    Returns:
        torch.Tensor: A 1D tensor containing all the labels in the dataset.
    """
    labels_list = []
    
    for _, labels in dataset:
        labels_list.append(labels.view(-1))  # Flatten the labels and store them
    
    # Concatenate all labels into a single tensor
    labels_tensor = torch.cat(labels_list)
    return labels_tensor

def plot_label_distribution_torch(labels_tensor, save_path="label_distribution.png"):
    filtered_labels = labels_tensor[labels_tensor != 0]
    # Count occurrences of each unique label
    unique_labels, counts = torch.unique(filtered_labels, return_counts=True)
    
    # Convert to CPU for plotting if necessary
    unique_labels = unique_labels.cpu().numpy()
    counts = counts.cpu().numpy()
    
    # Plotting the distribution
    plt.figure(figsize=(10, 6))
    plt.bar(unique_labels, counts, width=0.5, alpha=0.7, edgecolor='black')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title('Distribution of Labels')
    plt.xticks(unique_labels)  # Ensure all labels are shown on the x-axis
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot as an image
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as {save_path}")
    
    # Show the plot
    plt.show()
    
def weighted_tversky_loss(pred, target, alpha=0.8, beta=0.2, class_weights=None, smooth=1e-5):
    """
    Weighted Tversky loss for multiclass segmentation with flattened spatial dimensions.
    
    Arguments:
    - pred: Tensor of shape (N, D * H * W, 7) - model logits
    - target: Tensor of shape (N, D * H * W, 1) - ground truth labels
    - alpha: Controls the penalty for false negatives
    - beta: Controls the penalty for false positives
    - class_weights: Tensor of shape (7,) representing the weight of each class
    - smooth: Smoothing factor to avoid division by zero
    
    Returns:
    - loss: Weighted Tversky loss value
    """
    # Convert logits to probabilities using softmax
    pred = torch.softmax(pred, dim=-1)  # Apply along the class dimension (last dim)

    # One-hot encode the target labels to match the shape of predictions
    target_one_hot = torch.nn.functional.one_hot(target.squeeze(-1), num_classes=pred.shape[-1])
    target_one_hot = target_one_hot.float()  # Convert to float for computation

    # Calculate true positives, false negatives, and false positives
    true_pos = (pred * target_one_hot).sum(dim=1)  # Sum over spatial dimension (D * H * W)
    false_neg = ((1 - pred) * target_one_hot).sum(dim=1)
    false_pos = (pred * (1 - target_one_hot)).sum(dim=1)

    # Calculate Tversky index per class
    tversky_index = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)

    # Apply class weights (if provided)
    if class_weights is not None:
        class_weights = class_weights.to(pred.device)  # Ensure weights are on the same device
        tversky_index = tversky_index * class_weights

    # Return the weighted mean Tversky loss across classes and batches
    return (1 - tversky_index).mean()

def calculate_class_weights(labels, num_classes, device):
    # Flatten labels and count occurrences of each class
    flat_labels = labels.view(-1)
    counts = torch.bincount(flat_labels, minlength=num_classes)
    total = counts.sum().item()

    # Calculate weights as inverse frequency
    class_weights = total / (counts + 1e-5)
    class_weights /= class_weights.max()  # Normalize weights
    return class_weights.to(device)
    