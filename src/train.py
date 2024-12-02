import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from model import CNN
from torch.optim.lr_scheduler import StepLR

# Paths to the dataset
TRAIN_DIR = "data/asl_alphabet_train"
MODEL_PATH = "saved_models/sign_language_model.pth"

# Hyperparameters
BATCH_SIZE = 32  # Moderate batch size
LEARNING_RATE = 0.001
EPOCHS = 20  # Balanced epoch count

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training data with augmentation
def load_train_data(batch_size):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = ImageFolder(root=TRAIN_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, len(dataset.classes)

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs):
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # Halve LR every 5 epochs

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

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")

        # Epoch summary
        train_accuracy = 100.0 * correct / total
        train_loss = running_loss / len(train_loader)

        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%\n")

        scheduler.step()  # Adjust learning rate

# Main function
def main():
    print("Loading data...")
    train_loader, num_classes = load_train_data(BATCH_SIZE)

    print("Initializing model...")
    model = CNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # Add L2 regularization

    print("Training model...")
    train_model(model, train_loader, criterion, optimizer, EPOCHS)

    print(f"Saving model to {MODEL_PATH}...")
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model training complete!")

if __name__ == "__main__":
    main()