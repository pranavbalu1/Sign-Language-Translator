import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np
from src.model import CNN
from src.preprocess import load_and_preprocess_data
from sklearn.linear_model import LinearRegression

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
    
    # Predictions and Labels
    all_preds = []
    all_labels = []

    # Evaluate accuracy and collect predictions
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate precision, recall, F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    accuracy = np.mean(all_preds == all_labels)
    
    # Calculate mAP (mean Average Precision)
    report = classification_report(all_labels, all_preds, output_dict=True)
    mAP = np.mean([v['precision'] for k, v in report.items() if k.isdigit()])  # Average precision over all classes

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"mAP: {mAP:.4f}")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))

    # Visualize metrics
    visualize_metrics(precision, recall, f1, mAP)


def visualize_metrics(precision, recall, f1, mAP):
    num_classes = len(precision)
    classes = np.arange(num_classes)  # Numeric class indices for bar graph

    # Combine metrics for bar graph
    metrics = {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    plt.figure(figsize=(12, 8))
    
    bar_width = 0.2  # Width of each bar
    class_labels = [f"Class {i}" for i in classes]

    for idx, (metric_name, values) in enumerate(metrics.items()):
        # Calculate positions for bar placement
        bar_positions = classes + (idx - 1) * bar_width

        # Create bar graph
        plt.bar(bar_positions, values, width=bar_width, label=metric_name)

    # Add mAP line
    plt.axhline(mAP, color='gold', linestyle='--', label=f'mAP: {mAP:.2f}')

    # Add labels, title, and legend
    plt.xticks(classes, class_labels, rotation=45)
    plt.xlabel("Classes")
    plt.ylabel("Score")
    plt.title("Evaluation Metrics by Class")
    plt.legend()
    plt.tight_layout()
    plt.show()


