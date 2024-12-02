import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader, TensorDataset
from model import CNN

# Paths to datasets
AUGMENTED_TRAIN = "data/augmented/train_augmented.pkl"
VAL_DATA = "data/prepared/val.pkl"
MODEL_PATH = "saved_models/sign_language_model.pth"

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20

# Device configuration: Automatically use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
def load_data(path):
    with open(path, "rb") as f:
        images, labels = pickle.load(f)
    # Add a channel dimension to grayscale images
    if images.ndim == 3:  # If shape is [N, H, W], add channel dimension
        images = images[:, :, :, None]  # Add a channel dimension (C=1)
    # Convert to PyTorch tensors
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels

# Train function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move to CPU or GPU

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100.0 * correct / total
        train_loss = running_loss / len(train_loader)

        # Validate the model
        val_loss, val_accuracy = validate_model(model, val_loader, criterion)

        print(f"Epoch [{epoch + 1}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    return model

# Validate function
def validate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_accuracy = 100.0 * correct / total

    return val_loss, val_accuracy

# Main function
def main():
    # Load datasets
    print("Loading data...")
    train_images, train_labels = load_data(AUGMENTED_TRAIN)
    val_images, val_labels = load_data(VAL_DATA)

    # Create DataLoaders
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, loss, and optimizer
    print("Initializing model...")
    model = CNN().to(device)  # Move model to GPU or CPU
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    print("Training model...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)

    # Save the model
    print(f"Saving model to {MODEL_PATH}...")
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model training complete!")

if __name__ == "__main__":
    main()