import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import os

# Paths to the CSV files
TRAIN_CSV = "data/sign_mnist_train.csv"
TEST_CSV = "data/sign_mnist_test.csv"
OUTPUT_DIR = "data/prepared"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_data(csv_path):
    """Load and preprocess data from a CSV file."""
    # Load CSV file
    data = pd.read_csv(csv_path)
    
    # Separate features (pixel values) and labels
    labels = data["label"].values
    images = data.drop("label", axis=1).values

    # Normalize pixel values to range [0, 1]
    images = images / 255.0

    # Reshape images to (28, 28, 1) for CNN input
    images = images.reshape(-1, 28, 28, 1)

    return images, labels

def save_splits(train_data, val_data, test_data):
    """Save dataset splits as pickle files."""
    with open(os.path.join(OUTPUT_DIR, "train.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    with open(os.path.join(OUTPUT_DIR, "val.pkl"), "wb") as f:
        pickle.dump(val_data, f)
    with open(os.path.join(OUTPUT_DIR, "test.pkl"), "wb") as f:
        pickle.dump(test_data, f)

def main():
    # Preprocess training and testing datasets
    print("Preprocessing training data...")
    train_images, train_labels = preprocess_data(TRAIN_CSV)
    print("Preprocessing testing data...")
    test_images, test_labels = preprocess_data(TEST_CSV)

    # Split training data into train and validation sets
    print("Splitting data...")
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42
    )

    # Save the splits
    print("Saving splits...")
    save_splits(
        (train_images, train_labels),
        (val_images, val_labels),
        (test_images, test_labels)
    )
    print(f"Splits saved to {OUTPUT_DIR}.")

if __name__ == "__main__":
    main()