import torch
import matplotlib.pyplot as plt
from src.model import CNN
from PIL import Image
import numpy as np
from torchvision import transforms

def make_prediction(model_path, image, label):
    # Load model
    model = CNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Make prediction
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(1).item()
    
    # Display the image
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"True Label: {label}, Predicted: {prediction}")
    plt.show()


def predict_custom_image(model_path, image_path, class_mapping):
    # Load the model architecture
    model = CNN()  # Replace with the correct model class used during training
    model.load_state_dict(torch.load(model_path))  # Load state_dict into the model
    model.eval()

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convert to grayscale
        transforms.Resize((28, 28)),  # Resize to 28x28
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        output = model(image)
        _, predicted_label = torch.max(output, 1)
    
    # Map to class
    predicted_class = class_mapping[predicted_label.item()]
    print(f"Predicted Class: {predicted_class}")