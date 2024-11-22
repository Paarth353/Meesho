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
