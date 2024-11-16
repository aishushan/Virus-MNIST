# Virus-MNIST Classifiaction

### **Objective:**
The objective of this project is to build and evaluate a machine learning model for classifying viral images into different categories. The project focuses on leveraging a **convolutional neural network (CNN)** architecture, utilizing a pre-trained model (MobileNetV2) for image classification. The primary goal is to:

- **Process and prepare the dataset**: Clean the data and scale the images for training.
- **Build a robust model**: Utilize transfer learning with MobileNetV2 to take advantage of pre-trained features.
- **Evaluate model performance**: Optimize the model and assess its accuracy on a test set, aiming for high classification performance.

### **Goals:**
1. **Data Preprocessing**:
   - Load the dataset (train and test CSV files).
   - Handle missing or irrelevant features (e.g., dropping the 'hash' column).
   - Convert image data to a usable format for deep learning models (resizing, reshaping, and scaling images).
   - Transform grayscale images to RGB for the model's input.

2. **Model Development**:
   - Utilize MobileNetV2 (pre-trained) to leverage its ability to extract features from images.
   - Freeze the pre-trained layers to avoid re-training and use it as a feature extractor.
   - Add a custom dense layer for classification.
   - Compile the model with suitable loss function (Sparse Categorical Crossentropy) and optimizer (Adam).

3. **Model Training and Evaluation**:
   - Split the training dataset into a training and validation set.
   - Train the model on the training data while validating its performance on the validation set.
   - Evaluate the model on the test data to estimate its generalization capabilities.
   - Plot training and validation loss/accuracy curves to monitor performance.

4. **Model Improvement**:
   - Aim for the highest accuracy possible with adjustments in hyperparameters, layers, and training epochs.
   - Monitor model's performance by evaluating loss and accuracy on both training and test datasets.
   - Predict on a test dataset and print a few sample predictions alongside actual labels.

5. **Outcome**:
   - Achieve satisfactory accuracy, aiming for better than 59% as an initial result.
   - Display the model’s confusion matrix and evaluate misclassifications to identify areas for improvement.

### **Models Used**:
1. **MobileNetV2** (Pre-trained Model):
   - A deep convolutional neural network that has been pre-trained on ImageNet data and is capable of transferring learned features to the new task.
   - Used as a feature extractor by excluding its top layers (classification head) and adding custom dense layers for specific classification.
   - MobileNetV2 helps in reducing the model’s complexity and training time by leveraging transfer learning.

2. **Dense Layer** (Custom):
   - The output layer is designed with a **Dense layer** and **softmax** activation to classify the image into one of 10 categories. The model uses a **sigmoid** activation to output class probabilities for each image.

### **Working Process**:
1. **Data Loading**:
   - Load the training and test datasets from CSV files using `pandas.read_csv()`.
   - Inspect the shape and columns of the dataset.
   - Drop unnecessary columns such as `hash`.

2. **Data Preprocessing**:
   - **Resizing**: Each image is reshaped to a 32x32 dimension.
   - **Scaling**: The image pixel values are scaled between 0 and 1 by dividing by 255.0 to normalize them for neural network training.
   - **RGB Conversion**: Grayscale images are converted to 3-dimensional RGB images using `np.repeat()` to prepare the images for MobileNetV2 input.

3. **Model Creation**:
   - **Base Model (MobileNetV2)**: A pre-trained MobileNetV2 model is loaded with `weights='imagenet'` and `include_top=False` (no final classification layer).
   - **Global Average Pooling**: This step reduces the spatial dimensions of the output of MobileNetV2 and converts it into a 1D vector.
   - **Custom Output Layer**: A custom dense layer with 10 units (for 10 classes) is added on top, followed by a **sigmoid activation**.

4. **Model Compilation**:
   - **Optimizer**: Adam optimizer with a learning rate of 0.001.
   - **Loss Function**: `sparse_categorical_crossentropy` as it is a multi-class classification problem.
   - **Metrics**: Accuracy is used to track the performance of the model during training and evaluation.

5. **Training**:
   - Split the training data into training and validation sets using `train_test_split`.
   - Train the model for 10 epochs, monitoring the training and validation loss/accuracy for improvements.
   - Visualize the training and validation loss and accuracy curves to monitor the model’s progress.

6. **Evaluation**:
   - Evaluate the model on the test dataset to measure its final performance and accuracy.
   - Print predicted values alongside true labels for comparison.

7. **Results**:
   - After training, the model achieved a test accuracy of **59%**. It correctly predicted many samples, but also misclassified others, suggesting potential areas for improvement.
