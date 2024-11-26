import pandas as pd
from matplotlib.widgets import Button
import matplotlib.pyplot as plt

# Load the test dataset
test_path = "data/sign_mnist_test.csv"
test_data = pd.read_csv(test_path)

# Separate labels and pixel data
labels = test_data['label']
pixels = test_data.iloc[:, 1:]  # All columns except 'label'

# View the first few rows
print("Test Data Sample:")
print(test_data.head())

# Function to visualize a single test case
def visualize_test_case(index):
    """
    Visualizes a single test case from the dataset.
    Args:
        index (int): The index of the test case to visualize.
    """
    label = labels[index]
    image = pixels.iloc[index].values.reshape(28, 28)  # Reshape to 28x28
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()


# Initialize the index
index = [0]

def update_plot(event):
    if event.key == 'right':
        index[0] = (index[0] + 1) % len(labels)
    elif event.key == 'left':
        index[0] = (index[0] - 1) % len(labels)
    visualize_test_case(index[0])

def visualize_test_case(index):
    label = labels[index]
    image = pixels.iloc[index].values.reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.draw()

fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', update_plot)
visualize_test_case(index[0])
plt.show()
