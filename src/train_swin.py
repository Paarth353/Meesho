import os
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import copy
from preprocessing import categories, create_loaders
from custom_losses import multi_attribute_f1_loss
from timm import create_model  # Swin Transformer is in timm
from sklearn.decomposition import PCA
from timm.models.swin_transformer import SwinTransformer

weights_dir = "../weights/"
os.makedirs(weights_dir, exist_ok=True)
save_weights_path = os.path.join(weights_dir, "multi_output_model_best_Swin_base.pth")

# Specify the number of classes per attribute in each category
num_classes_list_per_category = [
    [4, 2, 2, 3, 2],  # Example: T-shirts
    [4, 6, 3, 8, 4, 3, 4, 5, 9, 2],  # Sarees
    [13, 2, 2, 2, 2, 2, 2, 3, 2],  # Kurtis
    [7, 3, 3, 3, 6, 3, 2, 2],  # Women T-shirts
    [12, 4, 2, 7, 2, 3, 6, 4, 4, 6],  # Women Tops & Tunics
]

class MultiOutputModel(nn.Module):
    def __init__(self, base_model, num_classes_list_per_category=num_classes_list_per_category, color_hist_bins=16):
        super(MultiOutputModel, self).__init__()
            
        # Shared feature extractor
        self.base_model = base_model
        
        self.gap = nn.AdaptiveAvgPool2d(1)  # Pool to a single spatial size (1x1)
    
        # Define category-specific heads
        self.category_heads = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            for _ in num_classes_list_per_category
        ])
        
        # Define attribute-specific output layers for each category
        self.attribute_heads = nn.ModuleList([
            nn.ModuleList([nn.Linear(128, num_classes) for num_classes in num_classes_list])
            for num_classes_list in num_classes_list_per_category
        ])

    def forward(self, x, category_idx):
        
        # Shared feature extraction
        x = self.base_model(x)
#         print(f"Shape after base model: {x.shape}")  # Expecting [batch_size, 1024, 7, 7]
        
        
        # Apply Global Average Pooling to reduce spatial size
        x = x.permute(0, 3, 1, 2)
        x = self.gap(x)  # Output size: [batch_size, channels, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten the output to [batch_size, channels]
        
        # Pass through category head
        x = self.category_heads[category_idx](x)
        # print(f"Shape after category head: {x.shape}")  # Expecting [batch_size, 128]

        # Obtain outputs for each attribute in the category
        outputs = [torch.softmax(head(x), dim=1) for head in self.attribute_heads[category_idx]]
        return outputs
    
def train_model(model, num_epochs=25):

    best_model_wts = copy.deepcopy(model.state_dict())
    
    # Initialize parameters for early stopping
    patience = 3  # Number of epochs to wait for improvement
    min_delta = 0  # Minimum improvement in loss to reset patience
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Set up ReduceLROnPlateau with a small delta for minimal improvement threshold
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.1,        # Factor to reduce the LR by
    patience=3,        # Number of epochs to wait after the last significant improvement
    threshold=0.001,   # Minimum change (delta) to qualify as an improvement
    threshold_mode='rel',  # Use 'rel' for relative threshold, so 0.001 means a 0.1% improvement
    min_lr=1e-6)       # Print when LR is reduced

    
    # Training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Lists to track losses and metrics
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        model.train()  # Set model to training mode
    
        # Track training loss per epoch
        epoch_train_loss = 0
        batch_count = 0
    
        i=0
        # Training phase
        for category_idx, category in enumerate(categories):
            print(f"\nTraining on category: {category['name']}")
            category_name = category['name']
            train_loader = train_loaders[category_name]  # Get train DataLoader for the category
    
            for batch_idx, (images, labels) in enumerate(train_loader):
                # Move data to device
                images = images.to(device)
                labels = labels.to(device)
    
                # Forward pass
                outputs = model(images, category_idx)
    
                        # Apply multi-attribute F1 loss with dynamic weights
                batch_loss = multi_attribute_f1_loss(
                    outputs, labels, category['num_attributes']
                )
                epoch_train_loss += batch_loss.item()
                batch_count += 1
    
                # Backpropagation and optimization
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                print(f"  [Batch {batch_idx+1}/{len(train_loader)}] Loss: {batch_loss.item():.4f}")
    
                # Print batch details
                # print(f"Category: {category_name} | Batch: {batch_idx+1}/{len(train_loader)} | Batch Loss: {batch_loss.item():.4f}")
    
            print(f"Category: {category_name} | Epoch {epoch+1}/{num_epochs}")
    
        
        # Calculate average training loss for the epoch
        avg_train_loss = epoch_train_loss / batch_count
        train_losses.append(avg_train_loss)
        print(f"Average Training Loss for Epoch {epoch+1}: {avg_train_loss:.4f}")
    
        # Validation phase
        model.eval()  # Set model to evaluation mode
        epoch_val_loss = 0
        val_batch_count = 0
        print("\nValidating...")
        with torch.no_grad():  # Disable gradient calculation
            for category_idx, category in enumerate(categories):
                category_name = category['name']
                val_loader = val_loaders[category_name]  # Get validation DataLoader for the category
                print(f"Validating on category: {category['name']}")
                attr_f1_scores = []
                for batch_idx, (images, labels) in enumerate(val_loader):
                    # Move data to device
                    images = images.to(device)
                    labels = labels.to(device)
    
                    # Forward pass
                    outputs = model(images, category_idx)
                    batch_loss = multi_attribute_f1_loss(outputs, labels, category['num_attributes'])
                    epoch_val_loss += batch_loss.item()
                    val_batch_count += 1
                    print(f"  [Batch {batch_idx+1}/{len(val_loader)}] Validation Loss: {batch_loss.item():.4f}")
                    
                print(f"Category: {category_name} | Epoch {epoch+1}/{num_epochs}")
            
        # Calculate average validation loss for the epoch
        avg_val_loss = epoch_val_loss / val_batch_count
        val_losses.append(avg_val_loss)
        print(f"Average Validation Loss for Epoch {epoch+1}: {avg_val_loss:.4f}")
        
        # Step the scheduler
        scheduler.step(avg_val_loss)
        
        # Early stopping check with min_delta
        if best_val_loss - avg_val_loss > min_delta:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0  # Reset counter if validation loss improved
            print(f"Best model updated with Validation Loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No significant improvement in validation loss for {epochs_no_improve} epoch(s)")
        
        # Check for early stopping
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break
    
      
    # After training, you could also plot the training and validation losses per epoch
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), save_weights_path)
    return model

if __name__ == "__main__":
    # Define transformations
    transform_Swin = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loaders, val_loaders, _ = create_loaders(batch_size=32, transform=transform_Swin)

    # Load the pre-trained Swin Transformer model without the classifier layer
    base_model = create_model('swin_base_patch4_window7_224', pretrained=True)

    # Define the model architecture
    # base_model = SwinTransformer(img_size=224, patch_size=4, window_size=7, embed_dim=128,
                                # depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), num_classes=0)

    # Load the weights
    # pretrained_weights = '../weights/swin_base.pth'
    # state_dict = torch.load(pretrained_weights, weights_only=True)

    # Update model weights
    # base_model.load_state_dict(state_dict)
    print("Pretrained weights loaded successfully!")


    base_model.head = nn.Identity()  # Remove the default classifier head

    # Freeze all layers first
    for param in base_model.parameters():
        param.requires_grad = False

    for name, param in base_model.named_parameters():
        if name.startswith(f"layers.3"):
            param.requires_grad = True


    # Number of blocks in stage 3 (index 2) to unfreeze
    num_blocks_to_unfreeze = 8  # Change this to the number of blocks you want to unfreeze
    # Get the total number of blocks in stage 3 based on the printed layer names
    # For instance, if stage 3 has 6 blocks, we would want the last 3 as an example.
    total_blocks_stage_3 = 18  # Update this based on the actual number you have
    # Calculate the starting block index to unfreeze
    start_block = total_blocks_stage_3 - num_blocks_to_unfreeze
    # Loop through the layers and set requires_grad appropriately
    for name, param in base_model.named_parameters():
        if any(name.startswith(f"layers.2.blocks.{start_block + i}") for i in range(num_blocks_to_unfreeze)):
            param.requires_grad = True

    num_classes_list_per_category = [[4,2,2,3,2],[4,6,3,8,4,3,4,5,9,2],[13,2,2,2,2,2,2,3,2],[7,3,3,3,6,3,2,2],[12,4,2,7,2,3,6,4,4,6]]

    # Create the model
    model_Swin = MultiOutputModel(base_model, num_classes_list_per_category)

    optimizer = optim.Adam(model_Swin.parameters(), lr=5.5e-5)

    # Calculate and print the number of trainable parameters
    def count_trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainable_params = count_trainable_params(model_Swin)
    print(f"Number of trainable parameters: {trainable_params}")

    trainable_params_swin = count_trainable_params(model_Swin.base_model)
    print(f"Number of trainable parameters in Swin base model: {trainable_params_swin}")

    model_ft = train_model(model_Swin, num_epochs=25)