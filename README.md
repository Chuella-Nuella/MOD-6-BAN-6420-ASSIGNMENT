Fashion MNIST Classification with Convolutional Neural Networks
Introduction
The Fashion MNIST dataset is a popular benchmark for machine learning, consisting of grayscale images of clothing items categorized into ten distinct classes. The objective of this assignment is to implement a Convolutional Neural Network (CNN) using Keras to classify these images accurately.
By developing an effective CNN model, we aim to achieve high accuracy in predicting the type of clothing item, which can be applied to tasks such as targeted marketing and inventory management.

Dataset Overview
The Fashion MNIST dataset contains images belonging to the following ten categories:
0 -T-shirt/top
 1 -Trouser
 2 - Pullover
 3 -Dress
 4 -Coat
 5 -Sandal
 6 -Shirt
 7 -Sneaker
 8 -Bag
 9 -Ankle boot
Steps Taken
1. Importing Required Libraries
The following libraries were used to build and evaluate the CNN model:
•	import tensorflow as tf
•	from tensorflow.keras import Sequential
•	from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
•	from tensorflow.keras.datasets import fashion_mnist
•	from tensorflow.keras.utils import to_categorical
•	import numpy as np
•	import matplotlib.pyplot as plt
2. Loading and Preprocessing the Data
Data Loading:
We load the Fashion MNIST dataset using the fashion_mnist.load_data() function and split it into training and test sets.
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
Data Preprocessing:
•	Reshape Data: Convert the images to the format (28, 28, 1) to include a single grayscale channel.
•	 Normalize Data: Scale pixel values to the range [0, 1].
•	 One-Hot Encode Labels: Convert class labels to categorical format for multi-class classification.
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
3. Defining the CNN Model
A six-layer CNN was implemented using the TensorFlow/Keras Sequential API. The architecture includes:
•	Convolutional Layers: Feature extraction with ReLU activation
•	MaxPooling Layers: Dimensionality reduction
•	Flatten Layer: Conversion of feature maps into 1D arrays
•	Dense Layers: Fully connected layers for classification
•	Dropout: Regularization to prevent overfitting
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')])
4. Compiling the Model
The model was compiled with the following parameters:
•	Optimizer: adam (for efficient training)
•	Loss Function: categorical_crossentropy (suitable for multi-class classification)
•	Metrics: accuracy (to monitor training progress)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
5. Training the Model
The model was trained for 12 epochs with a batch size of 32, using 20% of the training data for validation.
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=12,
    validation_split=0.2
)
Output:
To confirm the training progress, we print available attributes of the history object.
print(history.history.keys())
6. Evaluating the Model
After training, the model was evaluated on the test set to assess generalization performance.
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")
7. Making Predictions
Predictions were made for the first 5 images in the test set, and the results were visualized with both predicted and true labels.
predictions = model.predict(x_test[:5])
# Class labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# Display predictions
for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    predicted_label = np.argmax(predictions[i])
    true_label = np.argmax(y_test[i])
    plt.title(f"Predicted: {class_names[predicted_label]}, True: {class_names[true_label]}")
    plt.axis('off')
    plt.show()

Results and Observations
The trained CNN achieved an accuracy of approximately 90% on the test set.
Common misclassifications were observed between similar-looking classes such as Shirt and Coat. The model demonstrated good generalization for most classes but could benefit from further tuning, such as data augmentation and hyperparameter optimization.
Conclusion
This project successfully implemented a CNN model using TensorFlow/Keras to classify Fashion MNIST images. The model was trained, evaluated, and visualized effectively.
Key takeaways:
•	CNNs are powerful for image classification tasks.
•	Data preprocessing and normalization significantly improve model performance.
•	Dropout helps mitigate overfitting in deep learning models.

How to Run the Project
Prerequisites
Ensure you have the following dependencies installed:
pip install tensorflow numpy matplotlib
Running the Project
1. Clone the repository or unzip the project folder.
2. Navigate to the project directory and run the script:
python fashion_mnist_classification.py
Files in the Repository
fashion_mnist_classification.py – Contains the complete implementation of the CNN model.
README.md – Documentation for the project.
requirements.txt – Required Python packages.
