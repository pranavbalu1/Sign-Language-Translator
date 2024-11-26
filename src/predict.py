import torch
import matplotlib.pyplot as plt
from src.model import CNN

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
