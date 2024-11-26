import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(train_path=None, test_path=None):
    # Load test data if train_path is None
    if train_path:
        train_df = pd.read_csv(train_path)
        train_labels = torch.tensor(train_df['label'].values, dtype=torch.long)
        train_images = torch.tensor(train_df.drop('label', axis=1).values / 255.0, dtype=torch.float32)
        train_images = train_images.view(-1, 1, 28, 28)  # Reshape to (N, C, H, W)

        # Split into train and validation sets
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_images, train_labels, test_size=0.2, random_state=42
        )
    else:
        # Assign placeholders if train_path is None
        train_images, val_images, train_labels, val_labels = None, None, None, None

    if test_path:
        test_df = pd.read_csv(test_path)
        test_labels = torch.tensor(test_df['label'].values, dtype=torch.long)
        test_images = torch.tensor(test_df.drop('label', axis=1).values / 255.0, dtype=torch.float32)
        test_images = test_images.view(-1, 1, 28, 28)  # Reshape to (N, C, H, W)
    else:
        raise ValueError("Test path is required for evaluation or prediction.")

    return train_images, train_labels, val_images, val_labels, test_images, test_labels
