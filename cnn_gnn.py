"""IMU Character Recognition Pipeline

This module contains a compact, readable example pipeline to train a
multi-task model on IMU pen data. The script shows an end-to-end flow:

- Load pickled IMU and ground-truth data
- Normalize and pad variable-length sequences
- Extract features with a small CNN trunk
- Model temporal/relational structure with a simple GCN
- Combine into a multi-task model (classification + trajectory regression)

This file is intentionally straightforward to make it easy to read and
adapt. It is not organized as a library; it is a runnable example script.

Notes
- Expected data files: data/all_x_dat_imu.pkl, data/all_gt.pkl
- Keep changes minimal when extending; this file focuses on clarity.
"""

import os
import pickle
import logging
import time
from typing import List, Tuple

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from keras.models import Model, load_model
from keras.layers import (
    Conv1D, MaxPooling1D, Flatten, Dense, Input, Reshape, Dropout
)
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tabulate import tabulate

# Global variables to control behavior
DISPLAY_ALL_RESULTS = False
NUM_EPOCHS = 50
SHOW_NORMALIZED_SAMPLE = False
LOG_FILE = None  # Set to a file path to enable logging to a specific file
DISPLAY_TRAINING_PROGRESS = True

# Configure logging
if LOG_FILE:
    logging.basicConfig(
        filename=LOG_FILE, level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
else:
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
    )

# Helper function to log messages
def log_message(message: str) -> None:
    """Log a message to stdout and the configured logger.

    This keeps the previous behavior (print + logging.info) but makes the
    contract explicit via typing and a short docstring.
    """
    print(message)
    logging.info(message)

# File paths
imu_data_file = 'data/all_x_dat_imu.pkl'
gt_data_file = 'data/all_gt.pkl'
model_file = 'data/mtl_model.h5'

# Load the provided pkl files
start_time = time.time()
log_message("Loading IMU data and ground truth labels...")
# Load IMU data
with open(imu_data_file, 'rb') as f:
    imu_data = pickle.load(f)
log_message(f"IMU data loaded successfully. Number of samples: {len(imu_data)}")

# Load ground truth labels
with open(gt_data_file, 'rb') as f:
    gt_data = pickle.load(f)
log_message(f"Ground truth labels loaded successfully. Number of labels: {len(gt_data)}")

# Check if the number of IMU data samples matches the number of ground truth labels
if len(imu_data) != len(gt_data):
    raise ValueError(
        f"Mismatch in number of samples: IMU data has {len(imu_data)} samples, "
        f"but ground truth has {len(gt_data)} labels."
    )
log_message(f"Step 1: Data Loading completed in {time.time() - start_time:.2f} seconds")

# Convert ground truth labels to categorical format
start_time = time.time()
log_message("Converting ground truth labels to categorical format...")
# Convert labels (A-Z) to numerical format for training
gt_data_categorical = to_categorical([ord(label) - ord('A') for label in gt_data])
log_message(f"Ground truth labels converted successfully. Shape: {gt_data_categorical.shape}")
log_message(f"Step 1: Data Conversion completed in {time.time() - start_time:.2f} seconds")

# Step 1: Data Preprocessing
start_time = time.time()
log_message("Starting data preprocessing...")
log_message("Normalizing IMU data...")
scaler = MinMaxScaler()
# Normalize each sample using MinMaxScaler to bring all features into the same range
imu_data_normalized = [scaler.fit_transform(sample) for sample in imu_data]
if SHOW_NORMALIZED_SAMPLE:
    log_message(
        f"IMU data normalized successfully. Example normalized sample: "
        f"{imu_data_normalized[0]}"
    )

log_message("Padding IMU data sequences...")
# Pad sequences to make them uniform in length for model compatibility
imu_data_padded = pad_sequences(
    imu_data_normalized, padding='post', dtype='float32'
)
log_message(
    f"IMU data padded successfully. Data shape after padding: "
    f"{imu_data_padded.shape}"
)
log_message(f"Step 1: Data Preprocessing completed in {time.time() - start_time:.2f} seconds")

# Check if model file exists to avoid retraining
if os.path.exists(model_file):
    log_message("Loading existing MTL model from file...")
    mtl_model = load_model(model_file)
    log_message("Model loaded successfully.")
else:
    # Step 2: Feature Extraction using CNN
    start_time = time.time()
    log_message("Starting feature extraction using CNN...")
    # Input layer for CNN model
    input_layer = Input(shape=(imu_data_padded.shape[1], 13))
    # Add multiple 1D Convolutional layers to extract features from IMU data
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
    log_message(f"Conv1D layer 1 output shape: {x.shape}")
    x = MaxPooling1D(pool_size=2)(x)
    log_message(f"MaxPooling1D layer 1 output shape: {x.shape}")
    x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    log_message(f"Conv1D layer 2 output shape: {x.shape}")
    x = MaxPooling1D(pool_size=2)(x)
    log_message(f"MaxPooling1D layer 2 output shape: {x.shape}")
    # Dropout to reduce overfitting
    x = Dropout(0.3)(x)
    # Flatten the pooled feature maps into a single vector
    x = Flatten()(x)
    log_message(f"Flatten layer output shape: {x.shape}")
    # Dense layer to learn complex representations
    x = Dense(256, activation='relu')(x)
    log_message(f"Dense layer output shape: {x.shape}")
    log_message("Feature extraction using CNN completed.")
    log_message(f"Step 2: Feature Extraction completed in {time.time() - start_time:.2f} seconds")

    # Step 3: GNN for Relationship Modeling
    start_time = time.time()
    log_message("Creating GCN model for temporal relationships...")
    # Create edge indices for the GCN to define the connectivity between nodes
    edge_index = torch.tensor(
        [[i, i + 1] for i in range(imu_data_padded.shape[1] - 1)],
        dtype=torch.long
    ).t().contiguous()
    log_message(f"Edge index for GCN created. Edge index shape: {edge_index.shape}")
    # Create the input tensor for GCN from the first IMU data sample
    x_gnn = torch.tensor(imu_data_padded[0], dtype=torch.float)
    log_message(f"Input tensor for GCN created. Tensor shape: {x_gnn.shape}")

    # Create a Data object for the GCN
    data = Data(x=x_gnn, edge_index=edge_index)
    log_message("Data object for GCN created.")

    # Define the GCN model
    class GCN(torch.nn.Module):
        def __init__(self):
            super(GCN, self).__init__()
            # First GCN layer with 13 input features and 64 output features
            self.conv1 = GCNConv(13, 64)
            # Second GCN layer with 64 input features and 128 output features
            self.conv2 = GCNConv(64, 128)

        def forward(self, data):
            log_message("Running forward pass of GCN...")
            x, edge_index = data.x, data.edge_index
            # Apply first GCN layer and activation function
            x = self.conv1(x, edge_index).relu()
            log_message(f"GCNConv1 output shape: {x.shape}")
            # Apply second GCN layer
            x = self.conv2(x, edge_index)
            log_message(f"GCNConv2 output shape: {x.shape}")
            return x

    log_message("Instantiating GCN model...")
    # Instantiate and run the GCN model
    gnn_model = GCN()
    output_gnn = gnn_model(data)
    log_message(
        f"GCN model created and relationship modeling completed. "
        f"GCN output shape: {output_gnn.shape}"
    )
    log_message(f"Step 3: GNN Relationship Modeling completed in {time.time() - start_time:.2f} seconds")

    # Step 4: Multi-Task Learning (MTL)
    start_time = time.time()
    log_message("Creating MTL model with classification and regression heads...")
    # Classification head for predicting the character label
    classification_output = Dense(
        gt_data_categorical.shape[1], activation='softmax', name='classification'
    )(x)
    log_message(f"Classification head output shape: {classification_output.shape}")
    # Regression head for predicting the trajectory
    regression_output = Dense(
        imu_data_padded.shape[1] * 13, activation='linear'
    )(x)
    # Reshape regression output to match original IMU data dimensions
    regression_output = Reshape(
        (imu_data_padded.shape[1], 13), name='regression'
    )(regression_output)
    log_message(f"Regression head output shape after reshape: {regression_output.shape}")

    log_message("Building MTL model...")
    # Create the MTL model combining both classification and regression outputs
    mtl_model = Model(
        inputs=input_layer, outputs=[classification_output, regression_output]
    )
    log_message("MTL model created successfully.")

    # Compile and train the model
    log_message("Compiling and training the MTL model...")
    # Compile the model with appropriate loss functions for each task
    mtl_model.compile(
        optimizer='adam', loss=['categorical_crossentropy', 'mse'],
        metrics=['accuracy']
    )
    log_message("MTL model compiled successfully.")

    if DISPLAY_TRAINING_PROGRESS:
        log_message("Starting model training...")
    # Train the model on the IMU data and ground truth labels
    mtl_model.fit(
        imu_data_padded, [gt_data_categorical, imu_data_padded],
        epochs=NUM_EPOCHS, batch_size=32, verbose=DISPLAY_TRAINING_PROGRESS
    )
    log_message("MTL model training completed.")
    log_message(f"Step 4: MTL Model Training completed in {time.time() - start_time:.2f} seconds")

    # Save the trained model to a file
    mtl_model.save(model_file)
    log_message("MTL model saved to file.")

# Step 5: Displaying IMU Character, Ground Truth Character, and Accuracy in Table Format
start_time = time.time()
log_message("Evaluating model and displaying results...")
# Predict character labels from IMU data
predicted_labels, _ = mtl_model.predict(imu_data_padded)
# Convert predicted labels to character format
predicted_chars = [chr(np.argmax(pred) + ord('A')) for pred in predicted_labels]
# Ground truth characters
ground_truth_chars = [chr(ord(label)) for label in gt_data]

# Evaluate the model on the IMU data and get accuracy
evaluation = mtl_model.evaluate(
    imu_data_padded, [gt_data_categorical, imu_data_padded], verbose=0
)
classification_accuracy = evaluation[3]  # Accuracy of classification task

# Create a table with IMU character, ground truth character, and accuracy
table_data = []
mismatched_data = []
for i in range(len(predicted_chars)):
    if predicted_chars[i] != ground_truth_chars[i]:
        mismatched_data.append(
            [i + 1, predicted_chars[i], ground_truth_chars[i]]
        )
    if DISPLAY_ALL_RESULTS or predicted_chars[i] != ground_truth_chars[i]:
        table_data.append(
            [i + 1, predicted_chars[i], ground_truth_chars[i]]
        )

# Print the table if DISPLAY_ALL_RESULTS is True
if DISPLAY_ALL_RESULTS:
    log_message(
        tabulate(
            table_data,
            headers=["Sample #", "Predicted Character", "Ground Truth Character"],
            tablefmt="grid"
        )
    )
log_message(f'Classification Accuracy: {classification_accuracy * 100:.2f}%')

# Print mismatched characters
if mismatched_data:
    log_message("\nMismatched Characters:")
    log_message(
        tabulate(
            mismatched_data,
            headers=["Sample #", "Predicted Character", "Ground Truth Character"],
            tablefmt="grid"
        )
    )

    # Post-mortem analysis of mismatched characters
    log_message("\nPost-mortem Analysis of Mismatched Characters:")
    for sample in mismatched_data:
        sample_num, predicted_char, ground_truth_char = sample
        log_message(
            f"Sample {sample_num}: Predicted '{predicted_char}', but expected "
            f"'{ground_truth_char}'. Possible reasons could be model "
            f"overfitting, insufficient training data diversity, or difficulty "
            f"in distinguishing similar characters."
        )
else:
    log_message("\nNo mismatched characters found.")
