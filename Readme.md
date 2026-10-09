# 🌱 Plant Seedling Classification with Deep Learning

A computer vision project that classifies plant seedlings into 12 species using a convolutional neural network (CNN) and transfer learning with **ResNet-18**. The project includes a Streamlit web application for interactive image-based predictions.

## Overview

Identifying plant species at the seedling stage can be challenging because different plants often share similar visual characteristics. This project explores deep learning for multiclass seedling image classification.

Two approaches were developed and compared:

* **Custom CNN:** A convolutional neural network built with PyTorch.
* **ResNet-18:** A pretrained model adapted to classify 12 plant species using transfer learning and partial fine-tuning.

The trained model is integrated into a Streamlit application where users can upload an image and receive a predicted species label and model confidence score.

## Features

* Image classification across 12 plant species.
* Custom CNN baseline and pretrained ResNet-18.
* Transfer learning with selective fine-tuning.
* Image preprocessing with Torchvision.
* Interactive web interface built with Streamlit.
* Prediction confidence displayed as a percentage.

## Plant Species

The model predicts one of the following classes:

1. Black-grass
2. Charlock
3. Cleavers
4. Common Chickweed
5. Common wheat
6. Fat Hen
7. Loose Silky-bent
8. Maize
9. Scentless Mayweed
10. Shepherds Purse
11. Small-flowered Cranesbill
12. Sugar beet

## Tech Stack

* **Language:** Python
* **Deep learning:** PyTorch
* **Computer vision:** Torchvision, Pillow
* **Web application:** Streamlit
* **Data processing and evaluation:** NumPy, scikit-learn

## Model Development

### 1. Custom CNN

A convolutional neural network was implemented as a baseline to learn visual features from the seedling images.

### 2. Transfer Learning with ResNet-18

A pretrained ResNet-18 was adapted by replacing its final fully connected layer with a 12-class classification head. Earlier layers were frozen while the final residual block and classification head were fine-tuned.

The ResNet-18 model achieved a **weighted recall of 0.82** in the reported evaluation, compared with **0.41** for the custom CNN baseline.

Weighted recall accounts for the number of samples belonging to each class. These results reflect the evaluation performed during development; performance on external images has not yet been established.

## Project Structure

```text
seedling_classification/
├── app.py                  # Streamlit application
├── model_weights.pth       # Saved ResNet-18 weights
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

Training notebooks and the dataset may be maintained separately.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/seedling-classification.git
cd seedling-classification
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install torch torchvision streamlit pillow numpy scikit-learn
```

### 4. Add the trained model weights

Place `model_weights.pth` in the project directory, ensuring the filename matches the path used in `app.py`.

### 5. Run the application

```bash
streamlit run app.py
```

The application should open in your browser at `http://localhost:8501`.

## Usage

1. Launch the Streamlit application.
2. Upload a JPG, JPEG, or PNG image.
3. The application preprocesses the image and passes it to ResNet-18.
4. View the predicted species and the associated model confidence.

**Note:** The confidence score is the model's predicted probability, not a guarantee that the classification is correct. Images that differ significantly from the training data may be misclassified.

## Limitations

* The model is trained to select one of 12 known species and is not designed to reliably reject unknown plant types.
* Predictions may be affected by image quality, background, lighting, plant maturity, and differences between training and real-world images.
* Performance on external images requires further validation.
* A high confidence score does not necessarily imply a correct prediction.

## Future Improvements

* Evaluate per-class precision, recall, and F1-score using a held-out test set.
* Investigate misclassifications and external-image generalization.
* Add a display of probabilities for all 12 classes.
* Improve handling of out-of-distribution images.
* Deploy the application online.

## Acknowledgements

This project uses the Plant Seedlings Classification dataset and pretrained ResNet-18 weights provided through Torchvision.

