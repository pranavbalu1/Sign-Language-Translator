import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader, TensorDataset
from model import CNN

# Paths
MODEL_PATH = "saved_models/sign_language_model.pth"
TEST_DATA = "data/prepared/test.pkl"

# Device configuration: Automatically use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
def load_data(path):
    with open(path, "rb") as f:
        images, labels = pickle.load(f)
    # Add a channel dimension to grayscale images
    if images.ndim == 3:
        images = images[:, :, :, None]  # Add a channel dimension (C=1)
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels

# Evaluate function
def evaluate_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss / len(test_loader)
    test_accuracy = 100.0 * correct / total

    return test_loss, test_accuracy

# Main function
def main():
    # Load test data
    print("Loading test data...")
    test_images, test_labels = load_data(TEST_DATA)
    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load model
    print("Loading model...")
    model = CNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    print("Evaluating model...")
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()