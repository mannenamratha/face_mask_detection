# Face Mask Detection using CNN

## Overview
This project implements a Convolutional Neural Network (CNN) to classify images as either containing a face mask or not. The dataset is sourced from Kaggle and is preprocessed before training the deep learning model.

## Dependencies
To run this project, install the following dependencies:
```sh
!pip install kaggle
```

Additionally, the following Python libraries are used:
- NumPy
- Matplotlib
- OpenCV
- TensorFlow
- Scikit-learn
- PIL (Pillow)

## Dataset
The dataset is downloaded from Kaggle using the Kaggle API. It contains two classes:
- `with_mask`: Images of people wearing a face mask.
- `without_mask`: Images of people without a face mask.

### Downloading the Dataset
To download the dataset, use the following commands:
```sh
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d omkargurav/face-mask-dataset
```
The dataset is extracted using:
```python
from zipfile import ZipFile
with ZipFile('/content/face-mask-dataset.zip', 'r') as zip:
    zip.extractall()
    print('The dataset is extracted')
```

## Data Preprocessing
1. Read images from directories.
2. Assign labels (`1` for `with_mask` and `0` for `without_mask`).
3. Resize images to `128x128`.
4. Convert images to NumPy arrays.
5. Normalize pixel values by dividing by 255.
6. Split the dataset into training and testing sets (80-20 split).

## CNN Model Architecture
The model is built using TensorFlow and Keras:
- **Conv2D** layer with 32 filters (ReLU activation, 3x3 kernel)
- **MaxPooling2D** (2x2)
- **Conv2D** layer with 64 filters
- **MaxPooling2D**
- **Flatten** layer
- **Dense** layer with 128 neurons and Dropout (0.5)
- **Dense** layer with 64 neurons and Dropout (0.5)
- **Dense** output layer with 2 neurons (sigmoid activation)

### Compilation
```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
```

## Model Training
The model is trained for 10 epochs with a validation split of 10%.
```python
history = model.fit(X_train_scaled, Y_train, validation_split=0.1, epochs=10)
```

## Model Evaluation
The trained model is evaluated on the test dataset.
```python
loss, accuracy = model.evaluate(X_test_scaled, Y_test)
print('Test accuracy =', accuracy)
```

## Results
- The model achieved **94.11% accuracy** on the test dataset.
- Loss and accuracy plots indicate good model performance.

## Conclusion
This project successfully classifies face mask images using a CNN model with high accuracy. The approach can be further improved with data augmentation and hyperparameter tuning.

