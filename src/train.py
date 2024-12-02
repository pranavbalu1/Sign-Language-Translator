import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from model import CNN

# Paths to the dataset
TRAIN_DIR = "data/asl_alphabet_train"
MODEL_PATH = "saved_models/sign_language_model.pth"

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training data
def load_train_data(batch_size):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((28, 28)),                 # Resize to model input size
        transforms.ToTensor(),                       # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))         # Normalize (mean=0.5, std=0.5)
    ])
    dataset = ImageFolder(root=TRAIN_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, len(dataset.classes)

# Debug function to print sample data
def debug_dataloader(train_loader):
    print("Debugging DataLoader...")
    for images, labels in train_loader:
        print("Images shape:", images.shape)  # Should be (batch_size, 1, 28, 28)
        print("Labels:", labels[:10])         # First 10 labels
        break
    print("DataLoader debugging complete.\n")

# Debug function for model predictions
def debug_model(model, train_loader, criterion):
    print("Debugging Model...")
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        print("Model outputs:", outputs[:5])       # First 5 outputs
        print("Predicted labels:", outputs.argmax(1)[:5])  # First 5 predictions
        loss = criterion(outputs, labels)
        print("Loss:", loss.item())
        break
    print("Model debugging complete.\n")

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

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

            # Log every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")

        # Epoch summary
        train_accuracy = 100.0 * correct / total
        train_loss = running_loss / len(train_loader)

        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%\n")

# Main function
def main():
    print("Loading data...")
    train_loader, num_classes = load_train_data(BATCH_SIZE)
    debug_dataloader(train_loader)  # Debug dataset

    print("Initializing model...")
    model = CNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    debug_model(model, train_loader, criterion)  # Debug model predictions

    print("Training model...")
    train_model(model, train_loader, criterion, optimizer, EPOCHS)

    print(f"Saving model to {MODEL_PATH}...")
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model training complete!")

if __name__ == "__main__":
    main()