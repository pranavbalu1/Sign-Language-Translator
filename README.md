# **Sign Language Translator**

This project aims to create a **sign language translator** using computer vision and machine learning. It uses the **Sign Language MNIST** dataset to train a model that predicts hand gestures representing letters in American Sign Language (ASL). The project consists of several Python files, each handling different aspects of data preprocessing, model training, evaluation, and prediction.

## **Project Structure**

```bash
Sign-Language-Translator/
├── data/
│   ├── sign_mnist_train.csv
│   └── sign_mnist_test.csv
├── saved_models/
│   └── sign_language_model.pth
├── scripts/
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
└── main.py
```


## **Steps to Run**

1. **Install Dependencies**:
   Make sure all required libraries are installed using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**:
   Run the following command to start training:

   ```bash
   python main.py --mode train --train_path "data/sign_mnist_train.csv" --test_path "data/sign_mnist_test.csv" --model_path "saved_models/sign_language_model.pth"
   ```

3. **Evaluate the Model**:
   After training, evaluate the model's performance:

   ```bash
   python main.py --mode evaluate --test_path "data/sign_mnist_test.csv" --model_path "saved_models/sign_language_model.pth"
   ```

4. **Make Predictions**:
   To predict a specific image's label, run:

   ```bash
   python main.py --mode predict --test_path "data/sign_mnist_test.csv" --model_path "saved_models/sign_language_model.pth" --image_index 5
   ```


## **Description of Each File**

### **1. `main.py`**

The main entry point for running the project. It manages different operations like training, evaluation, and prediction based on the mode specified via command-line arguments.

#### **Methods in `main.py`:**

- `main()`: The main method that takes command-line arguments and calls the respective functionality (train, evaluate, or predict) based on the chosen mode.

#### **Customizations for `main.py`:**

- You can customize the `--mode` argument to select the operation (`train`, `evaluate`, or `predict`).
- Modify the `--train_path` and `--test_path` if you want to specify different datasets.
- Customize the `--model_path` to set a different location to save or load the model.

#### **How to Run**

1. **Training**:

   ```bash
   python main.py --mode train --train_path "data/sign_mnist_train.csv" --test_path "data/sign_mnist_test.csv" --model_path "saved_models/sign_language_model.pth"
   ```

2. **Evaluation**:

   ```bash
   python main.py --mode evaluate --test_path "data/sign_mnist_test.csv" --model_path "saved_models/sign_language_model.pth"
   ```

3. **Prediction**:

   ```bash
   python main.py --mode predict --test_path "data/sign_mnist_test.csv" --model_path "saved_models/sign_language_model.pth" --image_index 5
   ```

---

### **2. `scripts/preprocess.py`**

This file is responsible for loading and preprocessing the dataset. It reads the CSV files, scales the pixel values, and reshapes the data to be used by the model.

#### **Functions:**

- `load_and_preprocess_data(train_path=None, test_path=None)`: Loads and preprocesses the training and test data. The function will return the images and labels in the required format.
  - **train_path**: Path to the training dataset CSV file (used for training).
  - **test_path**: Path to the test dataset CSV file (used for evaluation/prediction).
  - **Returns**: Preprocessed training, validation, and test data.

#### **Customizations for `preprocess.py`:**

- You can change the `train_path` and `test_path` if your data is located elsewhere.
- If you want to use a different dataset format, modify the file parsing logic inside this function.

---

### **3. `scripts/model.py`**

This file defines the **Convolutional Neural Network (CNN)** model architecture. The model consists of convolutional layers followed by fully connected layers to classify hand gestures.

#### **Functions in `evaluate.py`:**

- `CNN`: Defines the CNN architecture, including convolution, pooling, and fully connected layers.

#### **Possible Customizations:**

- You can modify the number of layers or their parameters (e.g., kernel size, number of filters) in the `CNN` class to adjust the model complexity.
- Change the activation functions if needed (currently, ReLU is used).

---

### **4. `scripts/train.py`**

This file contains the code for training the model. It defines the loss function, optimizer, and the training loop.

#### **Functions in `train.py`:**

- `train_model(train_path, test_path, model_path)`: Trains the model on the training data and evaluates it on the test data. The trained model is then saved to `model_path`.
  - **train_path**: Path to the training dataset.
  - **test_path**: Path to the test dataset.
  - **model_path**: Path to save the trained model.

#### **Possible Customizations for `preprocess.py`:**

- Change the batch size or learning rate in the `train_model` function.
- You can also modify the number of epochs for training.

---

### **5. `scripts/evaluate.py`**

This file is responsible for evaluating the trained model on the test dataset. It calculates accuracy and other performance metrics.

#### **Functions in `predict.py`:**

- `evaluate_model(model_path, test_path)`: Loads the trained model from `model_path`, evaluates it on the test dataset, and prints the accuracy.
  - **model_path**: Path to the trained model.
  - **test_path**: Path to the test dataset.

#### **Possible Customizations for `evaluate.py`:**

- Modify the evaluation metrics (e.g., precision, recall) based on your requirements.

---

### **6. `scripts/predict.py`**

This file handles the task of predicting a hand gesture for a given test image. It takes an image and predicts the corresponding label.

#### **Methods:**

- `make_prediction(model_path, test_image, test_label)`: Loads the model and predicts the label for the provided image.
  - **model_path**: Path to the trained model.
  - **test_image**: The image to be predicted.
  - **test_label**: The true label of the image.

#### **Customizations:**

- You can adjust the `image_index` to predict different images.
- Modify the input format if you use a custom image format.

---

### **7. `scripts/utils.py`**

This is a utility file containing helper functions (if any are needed). It can be used to define functions that are used across multiple scripts.



---
