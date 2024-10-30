import numpy as np
from tqdm import trange
import struct
import matplotlib.pyplot as plt

def read_idx_file(file_path):
    with open(file_path, 'rb') as f:
        print(f"Reading file: {file_path}")
        magic_number = struct.unpack('>I', f.read(4))[0]
        num_items = struct.unpack('>I', f.read(4))[0]
        print(f"Number of Items: {num_items}")

        if 'images' in file_path:
            num_rows = struct.unpack('>I', f.read(4))[0]
            num_cols = struct.unpack('>I', f.read(4))[0]
            print(f"Image Dimensions: {num_rows}x{num_cols}")

            item_data = f.read()
            items = np.frombuffer(item_data, dtype=np.uint8)
            items = items.reshape(num_items, num_rows, num_cols)
        else:
            label_data = f.read()
            items = np.frombuffer(label_data, dtype=np.uint8)

        print(f"Items array shape: {items.shape}")
        print("-"*50)
        return items

# Load the MNIST dataset from IDX files
X_train = read_idx_file('train-images.idx3-ubyte__')
y_train = read_idx_file('train-labels.idx1-ubyte__')

# Binarize the images
def binarize_images(images):
    print("Binarizing images...")
    binarized_images = (images > 127).astype(int)
    return binarized_images.reshape(images.shape[0], -1)  # Flatten the images

X_train_binarized = binarize_images(X_train)

# Initialize parameters
def initialize_parameters(num_clusters, num_features):
    print("Initializing parameters...")
    np.random.seed(42)  # For reproducibility

    # Mixing coefficients (lambda)
    mixing_coefficients = np.full(num_clusters, 1 / num_clusters)

    # Bernoulli parameters (p)
    bernoulli_parameters = np.random.uniform(0.25, 0.75, (num_clusters, num_features))

    return mixing_coefficients, bernoulli_parameters

num_clusters = 10  # Digits 0-9
num_features = X_train_binarized.shape[1]

mixing_coefficients, bernoulli_parameters = initialize_parameters(num_clusters, num_features)

# E-step
def e_step(images, mixing_coefficients, bernoulli_parameters):
    num_samples = images.shape[0]
    num_clusters = mixing_coefficients.shape[0]
    responsibilities = np.zeros((num_samples, num_clusters))

    print("Performing E-step...")
    for k in trange(num_clusters):
        # To prevent underflow, compute log probabilities
        log_p_k = np.sum(
            images * np.log(bernoulli_parameters[k] + 1e-10) +
            (1 - images) * np.log(1 - bernoulli_parameters[k] + 1e-10),
            axis=1
        )
        log_responsibilities_k = np.log(mixing_coefficients[k] + 1e-10) + log_p_k
        responsibilities[:, k] = log_responsibilities_k

    # Normalize to get posterior probabilities
    max_log_responsibilities = np.max(responsibilities, axis=1, keepdims=True)
    responsibilities = np.exp(responsibilities - max_log_responsibilities)
    responsibilities_sum = np.sum(responsibilities, axis=1, keepdims=True)
    responsibilities /= responsibilities_sum

    return responsibilities

# M-step
def m_step(images, responsibilities):
    num_samples, num_features = images.shape
    num_clusters = responsibilities.shape[1]

    print("Performing M-step...")
    # Update mixing coefficients
    mixing_coefficients = np.sum(responsibilities, axis=0) / num_samples

    # Update Bernoulli parameters
    bernoulli_parameters = np.dot(responsibilities.T, images)
    cluster_weights = np.sum(responsibilities, axis=0).reshape(-1, 1)
    bernoulli_parameters /= cluster_weights + 1e-10  # Add small value to prevent division by zero

    return mixing_coefficients, bernoulli_parameters

# Convergence check
def has_converged(old_params, new_params, threshold=1e-2):
    difference = np.abs(new_params - old_params).sum()
    print(f"Convergence check: Total parameter change = {difference}")
    return difference < threshold

# EM algorithm
def run_em_algorithm(images, num_clusters=10, max_iterations=50):
    num_samples, num_features = images.shape
    mixing_coefficients, bernoulli_parameters = initialize_parameters(num_clusters, num_features)

    for iteration in range(1, max_iterations + 1):
        print(f"\nIteration {iteration}")
        # E-Step
        responsibilities = e_step(images, mixing_coefficients, bernoulli_parameters)

        # M-Step
        old_bernoulli_parameters = bernoulli_parameters.copy()
        mixing_coefficients, bernoulli_parameters = m_step(images, responsibilities)

        # Check for convergence
        if has_converged(old_bernoulli_parameters, bernoulli_parameters):
            print("Convergence reached.")
            break

    return mixing_coefficients, bernoulli_parameters, responsibilities

# Run EM algorithm
mixing_coefficients, bernoulli_parameters, responsibilities = run_em_algorithm(X_train_binarized)

# Assign clusters to labels
def assign_clusters(responsibilities, true_labels):
    num_clusters = responsibilities.shape[1]
    cluster_labels = np.zeros(num_clusters, dtype=int)

    print("Assigning clusters to labels...")
    for k in range(num_clusters):
        # Select data points with the highest responsibility for cluster k
        cluster_data_indices = np.argmax(responsibilities, axis=1) == k
        # Get the true labels of these data points
        true_labels_in_cluster = true_labels[cluster_data_indices]
        # Assign the cluster to the most frequent true label
        if len(true_labels_in_cluster) > 0:
            cluster_labels[k] = np.bincount(true_labels_in_cluster).argmax()
        else:
            cluster_labels[k] = -1  # If no data points assigned

    return cluster_labels

cluster_labels = assign_clusters(responsibilities, y_train)

# New function to print cluster imagination based on parameters
def print_imagination(p, mapping, num_row, num_col, labeled=False):
    for i in range(10):
        if labeled:
            print("Labeled", end=" ")
        print(f'Class {i}:')
        index = int(mapping[i])
        for row in range(num_row):
            for col in range(num_col):
                pixel = 1 if p[index][row * num_col + col] >= 0.5 else 0
                print(pixel, end=' ')
            print('')
        print('')

# Print the imagination of each cluster
print("\nImagination of each cluster:")
print_imagination(bernoulli_parameters, cluster_labels, 28, 28)

# Evaluate performance
def evaluate_performance(images, responsibilities, true_labels, cluster_labels):
    num_samples = images.shape[0]
    predicted_labels = np.zeros(num_samples, dtype=int)

    # Predict labels based on the highest responsibility
    assigned_clusters = np.argmax(responsibilities, axis=1)
    for i in range(num_samples):
        cluster = assigned_clusters[i]
        predicted_labels[i] = cluster_labels[cluster]

    # Calculate overall error rate
    error_rate = np.mean(predicted_labels != true_labels)
    print(f"\nTotal Error Rate: {error_rate}")

    # Calculate confusion matrices for each digit
    for digit in range(10):
        tp = np.sum((true_labels == digit) & (predicted_labels == digit))
        fn = np.sum((true_labels == digit) & (predicted_labels != digit))
        fp = np.sum((true_labels != digit) & (predicted_labels == digit))
        tn = np.sum((true_labels != digit) & (predicted_labels != digit))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        print(f"\nConfusion Matrix for Digit {digit}:")
        print(f"TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")
        print(f"Sensitivity: {sensitivity}")
        print(f"Specificity: {specificity}")

    return error_rate

error_rate = evaluate_performance(X_train_binarized, responsibilities, y_train, cluster_labels)
