import cv2
import torch
from torchvision import transforms
from PIL import Image
from model import CNN

# Paths
MODEL_PATH = "saved_models/sign_language_model.pth"
LABELS = [chr(i) for i in range(65, 91)] + ["del", "space", "nothing"]  # A-Z, delete, space, nothing

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
def load_model():
    model = CNN(num_classes=len(LABELS)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model

# Preprocessing for webcam frames
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return transform(pil_image).unsqueeze(0).to(device)

# Main function for webcam recognition
def main():
    model = load_model()

    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        input_tensor = preprocess_frame(frame)

        # Make a prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            predicted_label = LABELS[predicted.item()]

        # Display the prediction on the frame
        cv2.putText(frame, f"Prediction: {predicted_label}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Sign Language Translator", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()