import train_swin
import train_vgg
from preprocessing import categories, test_category_dfs, create_loaders

# Import PyTorch libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm  # For progress tracking
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from timm import create_model  # Swin Transformer is in timm
from timm.models.swin_transformer import SwinTransformer

# Check if any GPUs are available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

valid_path = '../data/raw/test_images'
sample = pd.read_csv('../data/raw/sample_submission.csv')

# Load the saved checkpoint
save_path_VGG = '../weights/multi_output_model_checkpoint_VGG_trained.pth'
save_path_Swin = '../weights/multi_output_model_checkpoint_Swin_trained.pth'

# Define the custom directory path for loading and saving weights
weights_dir = "../weights/"
os.makedirs(weights_dir, exist_ok=True)


# VGG
weights_path = os.path.join(weights_dir, "vgg19.pth")
# Load the base model with pretrained weights
base_model = models.vgg19()
base_model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
model_VGG = train_vgg.MultiOutputModel(base_model)
checkpoint = torch.load(save_path_VGG, weights_only=True, map_location=device)
model_VGG.load_state_dict(checkpoint['model_state_dict'])



# Swin
weights_path = os.path.join(weights_dir, "swin_base.pth")
# Define the model architecture
# base_model = SwinTransformer(img_size=224, patch_size=4, window_size=7, embed_dim=128,
#                             depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), num_classes=0)
# # Load the weights
# state_dict = torch.load(weights_path, weights_only=True, map_location=device)
# # Update model weights
# base_model.load_state_dict(state_dict)

base_model = create_model('swin_base_patch4_window7_224', pretrained=True)
base_model.head = nn.Identity()  # Remove the default classifier head
model_Swin = train_swin.MultiOutputModel(base_model)
checkpoint = torch.load(save_path_Swin, weights_only=True, map_location=device)
model_Swin.load_state_dict(checkpoint['model_state_dict'])


# Move model to the appropriate device if needed
model_Swin.to(device)
model_VGG.to(device)

categories = ['Men Tshirts', 'Sarees', 'Kurtis', 'Women Tshirts', 'Women Tops & Tunics']

# Define transformations for the test set
test_transforms_Swin = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_transforms_VGG = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

_, _, train_datasets = create_loaders(32, test_transforms_Swin)

# Define custom dataset for test images
class TestDataset(Dataset):
    def __init__(self, dataframe, directory, transform=None):
        self.dataframe = dataframe
        self.directory = directory
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_id = self.dataframe.iloc[idx]['id']
        img_path = f"{self.directory}/{img_id}"  # Assuming file extension .jpg
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return img_id, image


model_VGG.eval()
model_Swin.eval()

# Define weights for the models
vgg_weight = 0.8  # Weight for VGG model
swin_weight = 0.2  # Weight for Swin model

attributes_classes = {'Men Tshirts':[4,2,2,3,2],'Sarees':[4,6,3,8,4,3,4,5,9,2],\
                      'Kurtis':[13,2,2,2,2,2,2,3,2],\
                      'Women Tshirts':[7,3,3,3,6,3,2,2],'Women Tops & Tunics':[12,4,2,7,2,3,6,4,4,6]}


# Dictionary to store combined predictions across all subsets
combined_predictions = []

# Loop through each category in test_category_dfs and generate predictions
for subset_name, test_df in test_category_dfs.items():

    num_attributes = len(attributes_classes[subset_name])
    attributes = [f'attr_{i}' for i in range(1, num_attributes + 1)]
    
    # Load test images for the current category subset
    test_dataset_VGG = TestDataset(dataframe=test_df, directory=valid_path, transform=test_transforms_VGG)
    test_loader_VGG = DataLoader(test_dataset_VGG, batch_size=32, shuffle=False)

    # Load test images for the current category subset
    test_dataset_Swin = TestDataset(dataframe=test_df, directory=valid_path, transform=test_transforms_Swin)
    test_loader_Swin = DataLoader(test_dataset_Swin, batch_size=32, shuffle=False)

   # Initialize list for storing predictions for the current subset
    all_predictions = []

    with torch.no_grad():
        for (img_ids_vgg, images_vgg_v2), (img_ids_swin_v1, images_swin_v1) in tqdm(zip(test_loader_VGG, test_loader_Swin), desc=f"Generating predictions for {subset_name}", leave=False):
            
            
            images_vgg = images_vgg_v2.to(device)
            images_swin = images_swin_v1.to(device)
 
            # Get predictions from each model
            vgg_outputs = model_VGG(images_vgg_v2, categories.index(subset_name))
            swin_outputs = model_Swin(images_swin_v1, categories.index(subset_name))

            
            # Average the predictions with higher weight for the Swin model
            ensemble_outputs = [
                (vgg_weight * vgg_output + swin_weight * swin_output)
                for vgg_output, swin_output in zip(vgg_outputs, swin_outputs)
            ]
            

            # Gather predictions for all attributes in one pass
            batch_preds = [output.argmax(dim=1).cpu().numpy() for output in ensemble_outputs]
            all_predictions.append(np.stack(batch_preds, axis=1))  # Shape (batch_size, num_attributes)

    # Concatenate predictions to form a complete array for the subset
    predictions = np.concatenate(all_predictions, axis=0)

    # Decode predictions to match label mappings
    label_mappings = train_datasets[subset_name].get_label_mappings()
    def decode_predictions(predictions):
        decoded_labels = {}
        for i, attr in enumerate(attributes):
            decoded_labels[attr] = [label_mappings[attr][pred] for pred in predictions[:, i]]
        return decoded_labels

    decoded_results = decode_predictions(predictions)

    # Create DataFrame for the current subset
    subset_submission_df = pd.DataFrame(columns=sample.columns)
    for i, attr in enumerate(attributes, start=1):
        subset_submission_df[f'attr_{i}'] = decoded_results[attr]

    # Fill in the 'Category' and 'len' columns
    subset_submission_df['Category'] = subset_name
    subset_submission_df['len'] = len(attributes_classes[subset_name])

    # Copy the 'id' values from the test data DataFrame
    subset_submission_df['id'] = test_df['id'].values


    for i in range(num_attributes + 1, 11):
        subset_submission_df[f'attr_{i}'] = 'null'

    # Append the subset DataFrame to the combined predictions
    combined_predictions.append(subset_submission_df)

# Concatenate all category predictions into one final submission DataFrame
final_submission_df = pd.concat(combined_predictions, ignore_index=True)

final_submission_df['id'] = final_submission_df['id'].str.replace('.jpg', '', regex=False).astype(int)
# Replace NaN values with 'dummy_value'
# Replace 'null' with 'dummy_value' in the final DataFrame
final_submission_df.replace('null', 'dummy_value', inplace=True)

# Display the first few rows of the merged DataFrame
final_submission_df.to_csv("final_submission_10thNov2.csv", index=False)
final_submission_df
