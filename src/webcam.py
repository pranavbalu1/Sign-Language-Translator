import cv2
import torch
from torchvision import transforms
from PIL import Image
from model import CNN
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define label-to-letter mapping
label_to_letter = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
    10: "K", 11: "L", 12: "M", 13: "N", 14: "O",
    15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
    20: "U", 21: "V", 22: "W", 23: "X", 24: "Y"
}

# Load the trained model
model = CNN()
model.load_state_dict(torch.load("saved_models/sign_language_model.pth"))
model.eval()

# Define preprocessing for the webcam frames
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert OpenCV frame (numpy.ndarray) to PIL Image
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_pil = Image.fromarray(frame_gray)

    # Apply transformations
    frame_tensor = transform(frame_pil).unsqueeze(0)

    # Make a prediction
    with torch.no_grad():
        output = model(frame_tensor)
        _, predicted = torch.max(output, 1)
        predicted_label = predicted.item()
        predicted_sign = label_to_letter.get(predicted_label, "Unknown")

        # Calculate confidence
        probabilities = F.softmax(output, dim=1)
        confidence = torch.max(probabilities).item()

    # Display the prediction on the frame
    cv2.rectangle(frame, (5, 5), (300, 60), (0, 0, 0), -1)  # Black background
    cv2.putText(frame, f"Prediction: {predicted_sign} ({confidence:.2f})", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Sign Language Translator", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()