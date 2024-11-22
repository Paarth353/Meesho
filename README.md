# **Project Summary**

This project focuses on **multi-label classification** for a dataset containing products across **five categories**: *Men T-shirts, Sarees, Kurtis, Women T-shirts, and Women Tops & Tunics.* Each category has multiple attributes like **color, neck type, pattern, occasion**, etc., extracted from the accompanying parquet file.

Using an **ensemble learning approach** with **transfer learning**, the project leverages two pre-trained models:
- **VGG19**: Fine-tuned for feature extraction.
- **Swin Transformer**: A cutting-edge vision transformer model.

### **Core Innovations**
1. **Custom Soft F1 Loss Function**: Specifically tailored for imbalanced multi-attribute classification, ensuring robust performance across all attributes.
2. **Dynamic Preprocessing**: NaN values are strategically handled to optimize model learning and predictions.

### **Steps Involved**
1. **Preprocessing**: Cleaning and structuring data, managing missing values, and applying feature-specific transformations.
2. **Model Training**: Separate fine-tuning for VGG and Swin models, followed by **ensemble-based inference**.
3. **Ensemble Learning**: Weighted averaging of predictions from both models to boost accuracy and robustness.
4. **Evaluation**: Metrics include precision, recall, and F1-score for each category and attribute.

### **Real-World Applicability**
This solution is tailored for applications requiring **fine-grained categorization**, such as e-commerce product taxonomy, inventory tagging, and recommendation systems. It optimizes accuracy for multiple labels simultaneously across diverse product types.

This summary provides a concise yet comprehensive overview of your project. Let me know if you'd like further refinements!

# Setup Instructions for Meesho Multi-Attribute Classification Project

This document provides step-by-step instructions to set up and run the Meesho project.

---

## Recap of Directory Purpose
1. **`data/`:** Placeholder for datasets (link to download instead of uploading).
2. **`src/`:** Contains scripts for preprocessing, training, inference, and utilities.
3. **`weights/`:** Stores pretrained and trained model weights (upload links for large files).
4. **`submission/`:** Stores `submission.csv` created during inference.
5. **`notebooks/`:** (Optional) For exploratory analysis or visualization.

## Below are the steps for setting up and running the mode:
## 1. Clone the Repository

Start by cloning the repository to your local machine:
```bash
git clone https://github.com/Paarth353/Meesho.git
cd Meesho
```

## 2. Install Dependencies 
Install the required Python packages by running:
```bash
# Create a virtual environment
python -m venv env
source env/bin/activate   # For Linux/Mac
.\env\Scripts\activate      # For Windows

# Install dependencies
pip install -r requirements.txt
```

## 3. Download the dataset:
Kaggle: Download the dataset from this link [Kaggle](https://www.kaggle.com/competitions/visual-taxonomy/data) 
G-Drive: Download the dataset from this link [G-Drive](https://drive.google.com/drive/folders/1wTAHyzmMs51ypUBEo6d7Jgsby-lg24_a?usp=sharing)
After downloading, place the dataset files in the data/raw/ directory

## 4. Preprocess the dataset:
Preprocess the raw data and generate training-ready files:
```bash
python src/feature_engineering.py
```

## 5. Download Pretrained Weights:
Download the pretrained and trained model weights by following the instructions given in the **`weights/`:** directory to organise the data

## 6. Train the Model:
Start training the model using the preprocessed dataset:
```bash
# Train VGG model
python src/train_vgg.py

# Train Swin Transformer model
python src/train_swin.py
```

## 7. Evaluate the Model(Inference):
You can infer the model using the pretrained weights provided in the **`weights/`:** directory or you can train and store the weights in the same for new evaluation:
```bash
# Generate predictions and create submission.csv
python src/ensemble_inference.py
```
## 8. Use the generated submission.csv file:
The python file given above will generate the required final **`submission.csv`** file which can be further used for evaluation
