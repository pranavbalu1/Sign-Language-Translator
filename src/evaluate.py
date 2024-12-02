import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from model import CNN

# Paths
TRAIN_DIR = "data/asl_alphabet_train"
MODEL_PATH = "saved_models/sign_language_model.pth"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load validation/test data from the training dataset
def load_split_data(batch_size, split_ratio=0.2):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((28, 28)),                 # Resize to model input size
        transforms.ToTensor(),                       # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))         # Normalize
    ])
    dataset = ImageFolder(root=TRAIN_DIR, transform=transform)

    # Split dataset into training and validation sets
    val_size = int(len(dataset) * split_ratio)
    train_size = len(dataset) - val_size
    _, val_dataset = random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return val_loader, len(dataset.classes)

# Evaluate function
def evaluate_model(model, val_loader, criterion):
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
    print("Loading validation data...")
    val_loader, num_classes = load_split_data(batch_size=64)

    print("Loading model...")
    model = CNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))

    criterion = nn.CrossEntropyLoss()

    print("Evaluating model...")
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

if __name__ == "__main__":
    main()