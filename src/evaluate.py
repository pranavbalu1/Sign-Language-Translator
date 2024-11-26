import torch
from torch.utils.data import DataLoader, TensorDataset
from src.model import CNN
from src.preprocess import load_and_preprocess_data

def evaluate_model(model_path, test_path):
    # Load test data
    _, _, _, _, test_images, test_labels = load_and_preprocess_data(None, test_path)
    
    # Create DataLoader
    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Load the model
    model = CNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Evaluate accuracy
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
    
    accuracy = correct / len(test_dataset)
    print("Test Accuracy:", accuracy)
