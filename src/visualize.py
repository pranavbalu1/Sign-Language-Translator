import pandas as pd
from matplotlib.widgets import Button
import matplotlib.pyplot as plt

# Load the test dataset
test_path = "data/sign_mnist_test.csv"
test_data = pd.read_csv(test_path)

# Separate labels and pixel data
labels = test_data['label']
pixels = test_data.iloc[:, 1:]  # All columns except 'label'

# Create a dictionary for mapping labels to letters
# Assuming labels 0-25 correspond to A-Z (excluding 'J' and 'Z' as they require motion)
label_to_letter = {i: chr(65 + i) for i in range(26)}  # 65 is the ASCII value of 'A'

# Initialize the index
index = [0]

# Function to visualize a single test case
def visualize_test_case(index):
    """
    Visualizes a single test case from the dataset.
    Args:
        index (int): The index of the test case to visualize.
    """
    numeric_label = labels[index]
    letter_label = label_to_letter.get(numeric_label, "?")  # Use '?' for undefined labels
    image = pixels.iloc[index].values.reshape(28, 28)  # Reshape to 28x28
    
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {numeric_label} (Letter: {letter_label})")
    plt.axis('off')
    plt.draw()

# Event handler for updating the plot
def update_plot(event):
    if event.key == 'right':
        index[0] = (index[0] + 1) % len(labels)
    elif event.key == 'left':
        index[0] = (index[0] - 1) % len(labels)
    plt.clf()  # Clear the previous plot
    visualize_test_case(index[0])
    plt.pause(0.1)

# Set up the figure and connect the event
fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', update_plot)
visualize_test_case(index[0])
plt.show()