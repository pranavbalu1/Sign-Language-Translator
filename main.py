import argparse
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import make_prediction
from src.preprocess import load_and_preprocess_data

def main():
    parser = argparse.ArgumentParser(description="Sign Language Recognition Pipeline")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'predict'],
                        help="Choose the mode: 'train', 'evaluate', or 'predict'")
    parser.add_argument('--train_path', type=str, default="data/sign_mnist_train.csv",
                        help="Path to the training dataset (for training).")
    parser.add_argument('--test_path', type=str, default="data/sign_mnist_test.csv",
                        help="Path to the test dataset (for evaluation or prediction).")
    parser.add_argument('--model_path', type=str, default="saved_models/sign_language_model.pth",
                        help="Path to save or load the model.")
    parser.add_argument('--image_index', type=int, default=0,
                        help="Index of the test image to use for prediction (only for 'predict' mode).")
    args = parser.parse_args()

    if args.mode == 'train':
        print("Starting training...")
        train_model(args.train_path, args.test_path, args.model_path)
        print("Training completed.")

    elif args.mode == 'evaluate':
        print("Starting evaluation...")
        evaluate_model(args.model_path, args.test_path)
        print("Evaluation completed.")

    elif args.mode == 'predict':
        print("Starting prediction...")
        # Load test data and preprocess it
        _, _, _, _, test_images, test_labels = load_and_preprocess_data(None, args.test_path)
        # Use the specified image index for prediction
        test_image = test_images[args.image_index]
        test_label = test_labels[args.image_index].item()
        make_prediction(args.model_path, test_image, test_label)
        print("Prediction completed.")

if __name__ == "__main__":
    main()
