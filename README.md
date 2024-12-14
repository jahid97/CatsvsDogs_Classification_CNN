# 🐾Cats vs Dogs Classification using CNN

This project implements a **Convolutional Neural Network (CNN)** to classify images of cats and dogs using the popular "Dogs vs Cats" dataset from Kaggle.

---

##🌟Overview
The goal of this project is to classify images as either "cat" or "dog" using a deep learning model. A CNN is built and trained using TensorFlow/Keras on the Dogs vs Cats dataset.

### Project Overview
This project leverages a Convolutional Neural Network (CNN) to analyze and classify images into two categories: cats and dogs. The workflow involves:
- **Dataset Acquisition**: Using the Kaggle API to download the "Dogs vs Cats" dataset.
- **Data Preprocessing**: Resizing images, data augmentation, and normalization for efficient model training.
- **Model Development**: Building a CNN architecture using TensorFlow/Keras optimized for binary image classification.
- **Training and Evaluation**: Training the model on the dataset and evaluating its performance using accuracy and loss metrics.
- **Prediction**: Testing the trained model on unseen images to predict whether they contain a cat or dog.

This project is ideal for beginners to intermediate-level practitioners looking to understand image classification and CNN fundamentals.

Key highlights:
- Dataset processing and augmentation
- Model training with CNN architecture
- Evaluation of model performance

## Dataset
The dataset used is the **Dogs vs Cats** dataset from Kaggle:

- **Link**: [Dogs vs Cats on Kaggle](https://www.kaggle.com/salader/dogs-vs-cats)
- **Size**: ~25,000 images (12,500 cats and 12,500 dogs)

> Note: You need Kaggle API access to download the dataset programmatically.

## Requirements
Make sure the following dependencies are installed:

- Python 3.x
- TensorFlow/Keras
- NumPy
- Matplotlib
- Kaggle API

To install the required libraries, run:
```bash
pip install tensorflow numpy matplotlib kaggle
```

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd CatsvsDogs_prediction_CNN-main
   ```

2. Download the dataset:
   - Place your Kaggle API key (`kaggle.json`) in the appropriate directory.
   - Run the notebook or use the following command to download the dataset:
     ```bash
     kaggle datasets download -d salader/dogs-vs-cats
     ```

3. Extract the dataset:
   ```python
   import zipfile
   zip_ref = zipfile.ZipFile("dogs-vs-cats.zip", 'r')
   zip_ref.extractall("./data")
   zip_ref.close()
   ```

4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook CatsvsDogs_prediction_CNN.ipynb
   ```

## Results
After training the model, you can expect the following performance (example results):

- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~90%

Sample predictions:
| Input Image | Predicted Label |
|-------------|-----------------|
| ![Cat](example_cat.jpg) | Cat |
| ![Dog](example_dog.jpg) | Dog |

## Contributing
Contributions are welcome! If you have suggestions or improvements, please submit a pull request.

---

## Acknowledgment
Special thanks to **Kaggle** for providing the Dogs vs Cats dataset and to the **TensorFlow** and **Keras** teams for their powerful deep learning libraries.

