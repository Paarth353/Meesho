# datasets
import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Define the custom Dataset class to handle multiple attributes
class MultiAttributeDataset(Dataset):
    
    def __init__(self, dataframe, img_dir, attributes, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.attributes = attributes
        self.transform = transform
        
        # Initialize LabelEncoders for each attribute
        self.label_encoders = {attr: LabelEncoder() for attr in attributes}
        # Fit the encoders on the unique labels from the DataFrame
        for attr in attributes:
            valid_labels = self.dataframe[attr].dropna().unique() 
            self.label_encoders[attr].fit(valid_labels)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        
        # Load the image
        img_id = self.dataframe.iloc[idx]['id']
        img_path = f"{self.img_dir}/{img_id}"  # Adjust based on your file format
        image = Image.open(img_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Load the attribute labels and convert them to tensor format
        labels = []
        for attr in self.attributes:
            label = self.dataframe.iloc[idx][attr]
            if pd.isna(label):
                # Assign a default value (e.g., -1) for NaNs to ignore during loss calculation
                encoded_label = -1
            else:
                encoded_label = self.label_encoders[attr].transform([label])[0]
            labels.append(encoded_label)

        
        # Convert labels to tensor
        labels = torch.tensor(labels, dtype=torch.long)
        return image, labels

    def get_label_mappings(self):

        # Returns a dictionary of mappings from encoded labels to actual labels for each attribute
        return {attr: self.label_encoders[attr].classes_ for attr in self.attributes}

def to_jpg(id):

    # Format the ID as a zero-padded 6-digit string and add .jpg if not already present
    formatted_id = f"{int(id):06}"
    
    if not formatted_id.endswith(".jpg"):
        formatted_id += ".jpg"
    return formatted_id


# In preprocessing.py
def create_loaders(batch_size, transform):
    train_loaders = {}
    val_loaders = {}
    train_datasets = {}


    # Loop through each category and create DataLoaders
    for category in categories:
        name = category['name']
        num_attributes = category['num_attributes']
        dataframe = category['dataframe']
        img_dir = category['img_dir']

        # Define attribute names dynamically based on num_attributes
        attributes = [f'attr_{i}' for i in range(1, num_attributes + 1)]

        # Split the DataFrame into training and validation
        train_df = dataframe.iloc[:int(0.8 * len(dataframe))].reset_index(drop=True)
        val_df = dataframe.iloc[int(0.8 * len(dataframe)):].reset_index(drop=True)

        # Create the dataset instances
        train_dataset = MultiAttributeDataset(train_df, img_dir, attributes, transform=transform)
        val_dataset = MultiAttributeDataset(val_df, img_dir, attributes, transform=transform)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        )

        # Store DataLoaders in dictionaries for easy access
        train_loaders[name] = train_loader
        val_loaders[name] = val_loader
        train_datasets[name] = train_dataset

    return train_loaders, val_loaders, train_datasets


# Define paths relative to the `src` directory
raw_data_path = os.path.join("..", "data", "raw")
processed_data_path = os.path.join("..", "data", "processed")

# Ensure the processed directory exists
os.makedirs(processed_data_path, exist_ok=True)

# Load data files from the raw data directory
labels = pd.read_csv(os.path.join(raw_data_path, "train.csv"))
testing = pd.read_csv(os.path.join(raw_data_path, "test.csv"))
sample = pd.read_csv(os.path.join(raw_data_path, "sample_submission.csv"))
parquet = pd.read_parquet(os.path.join(raw_data_path, "category_attributes.parquet"))

train_path = os.path.join(raw_data_path, "train_images")
valid_path = os.path.join(raw_data_path, "test_images")

# Apply the function to both DataFrames
labels['id'] = labels['id'].apply(to_jpg)
sample['id'] = sample['id'].apply(to_jpg)
testing['id'] = testing['id'].apply(to_jpg)

# Check the unique categories in the Category column
categories = labels["Category"].unique()

# Create a dictionary to hold DataFrames for each category
category_dfs = {}

# Iterate through each category and filter the rows accordingly
for category in categories:
    category_df = labels[labels["Category"] == category]
    category_dfs[category] = category_df  # Store each DataFrame in the dictionary

    # Optionally, save each category DataFrame as a separate CSV file
    category_df.to_csv(os.path.join(processed_data_path, f"{category.replace(' ', '_')}_dataset.csv"), index=False)

# Create a dictionary to hold DataFrames for each category
test_category_dfs = {}

# Iterate through each category and filter the rows accordingly
for category in categories:
    test_category_df = testing[testing["Category"] == category]
    test_category_dfs[category] = test_category_df  # Store each DataFrame in the dictionary

    # Optionally, save each category DataFrame as a separate CSV file
    test_category_df.to_csv(os.path.join(processed_data_path, f"{category.replace(' ', '_')}_test_dataset.csv"), index=False)

attributes_classes = {'Men Tshirts':[4,2,2,3,2],'Sarees':[4,6,3,8,4,3,4,5,9,2],\
                      'Kurtis':[13,2,2,2,2,2,2,3,2],\
                      'Women Tshirts':[7,3,3,3,6,3,2,2],'Women Tops & Tunics':[12,4,2,7,2,3,6,4,4,6]}

for name in categories:
    num_attributes = len(attributes_classes[name])
    for i in range(1, num_attributes + 1):
        attr_column = f'attr_{i}'
        # Use .loc to avoid the SettingWithCopyWarning
        category_dfs[name].loc[:, attr_column] = category_dfs[name][attr_column].fillna(np.nan)


# List or dictionary of categories, each with its specific attributes and data paths
categories = [
    {'name': 'Men Tshirts', 'num_attributes': 5, 'dataframe': category_dfs['Men Tshirts'], 'img_dir': train_path},
    {'name': 'Sarees', 'num_attributes': 10, 'dataframe': category_dfs['Sarees'], 'img_dir': train_path},
    {'name': 'Kurtis', 'num_attributes': 9, 'dataframe': category_dfs['Kurtis'], 'img_dir': train_path},
    {'name': 'Women Tshirts', 'num_attributes': 8, 'dataframe': category_dfs['Women Tshirts'], 'img_dir': train_path},
    {'name': 'Women Tops & Tunics', 'num_attributes': 10, 'dataframe': category_dfs['Women Tops & Tunics'], 'img_dir': train_path}
]

