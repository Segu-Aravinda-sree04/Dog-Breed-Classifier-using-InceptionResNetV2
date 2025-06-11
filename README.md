# 🐶 Dog Breed Classifier using InceptionResNetV2

This project implements a dog breed classification model using transfer learning with InceptionResNetV2. It identifies over 100 different dog breeds from images, trained on the Kaggle Dog Breed Identification dataset.

## 🚀 Features

- Uses **InceptionResNetV2** as a feature extractor (bottleneck features)
- Custom fully connected neural network for classification
- High-resolution image preprocessing (299x299)
- Handles class imbalance and overfitting using:
  - Dropout layers
  - Early stopping
- Visualizes model performance using accuracy and loss plots
- Includes image prediction pipeline for single-image inference

## 🧠 Tech Stack

- Python
- TensorFlow / Keras
- InceptionResNetV2
- NumPy, Pandas, Matplotlib, Seaborn
- OpenCV & PIL for image manipulation

## 📊 Dataset

- [Dataset](https://www.kaggle.com/competitions/dog-breed-identification/data)
- Contains over 10,000 labeled dog images across 120 breeds

## 🔁 Model Architecture

- Bottleneck features from InceptionResNetV2
- Flattened and passed through:
  - Dense Layer (512) + ReLU + Dropout(0.2)
  - Dense Layer (512) + ReLU + Dropout(0.2)
  - Final Dense Layer with Softmax for breed classification

## 🖼️ Example Output

- **Training and Validation Accuracy**

![training and validation accuracy](https://github.com/user-attachments/assets/b877204c-c6be-4808-801c-397e47a23540)

- **Training and Validation Loss**

![training and validation loss](https://github.com/user-attachments/assets/5694f823-51a2-43a3-9bb9-d7a5c76dbe99)


## 📈 Results

- Achieved robust performance on training and validation datasets.

