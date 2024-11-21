import os
import torch
import torchvision.models as models
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import copy
from preprocessing import categories, create_loaders
from custom_losses import multi_attribute_f1_loss

# Define the custom directory path for loading and saving weights
weights_dir = "../weights/"
os.makedirs(weights_dir, exist_ok=True)
weights_path = os.path.join(weights_dir, "vgg19.pth")
save_weights_path = os.path.join(weights_dir, "multi_output_model_best_VGG19.pth")

# Specify the number of classes per attribute in each category
num_classes_list_per_category = [
    [4, 2, 2, 3, 2],  # Example: T-shirts
    [4, 6, 3, 8, 4, 3, 4, 5, 9, 2],  # Sarees
    [13, 2, 2, 2, 2, 2, 2, 3, 2],  # Kurtis
    [7, 3, 3, 3, 6, 3, 2, 2],  # Women T-shirts
    [12, 4, 2, 7, 2, 3, 6, 4, 4, 6],  # Women Tops & Tunics
]

# Define the MultiOutputModel
class MultiOutputModel(nn.Module):
    def __init__(self, base_model, num_classes_list_per_category=num_classes_list_per_category):
        super(MultiOutputModel, self).__init__()
        
        # Shared feature extractor
        self.base_model = base_model.features
        
        # Define category-specific heads
        self.category_heads = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(32768, 256),
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
        
        # Route to the appropriate category head
        x = self.category_heads[category_idx](x)
        
        # Obtain outputs for each attribute in the category
        outputs = [torch.softmax(head(x), dim=1) for head in self.attribute_heads[category_idx]]
        return outputs


# Define the training function
def train_model(model, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 3
    min_delta = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, threshold=0.001,
        threshold_mode='rel', min_lr=1e-6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)

        model.train()
        epoch_train_loss = 0
        batch_count = 0

        for category_idx, category in enumerate(categories):
            print(f"\nTraining on category: {category['name']}")
            batch_idx=0
            train_loader = train_loaders[category['name']]
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, category_idx)
                batch_loss = multi_attribute_f1_loss(outputs, labels, category['num_attributes'])
                epoch_train_loss += batch_loss.item()
                batch_count += 1
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                # Print progress every 10 batches
                # if (batch_idx + 1) % 10 == 0:
                print(f"  [Batch {batch_idx+1}/{len(train_loader)}] Loss: {batch_loss.item():.4f}")

                batch_idx+=1

        avg_train_loss = epoch_train_loss / batch_count
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0
        val_batch_count = 0

        print("\nValidating...")
        with torch.no_grad():
            for category_idx, category in enumerate(categories):
                print(f"Validating on category: {category['name']}")
                val_loader = val_loaders[category['name']]
                batch_idx=0
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images, category_idx)
                    batch_loss = multi_attribute_f1_loss(outputs, labels, category['num_attributes'])
                    epoch_val_loss += batch_loss.item()
                    val_batch_count += 1
                    # Print progress every 10 batches
                    if (batch_idx + 1) % 10 == 0:
                        print(f"  [Batch {batch_idx+1}/{len(val_loader)}] Validation Loss: {batch_loss.item():.4f}")
                    batch_idx+=1

        avg_val_loss = epoch_val_loss / val_batch_count
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        if best_val_loss - avg_val_loss > min_delta:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"Best model updated with Validation Loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.show()

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), save_weights_path)
    return model


if __name__ == "__main__":

    # Define transformations
    transform_VGG = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loaders, val_loaders, _ = create_loaders(batch_size=32, transform=transform_VGG)

    # Specify the number of classes per attribute in each category
    num_classes_list_per_category = [
        [4, 2, 2, 3, 2],  # Example: T-shirts
        [4, 6, 3, 8, 4, 3, 4, 5, 9, 2],  # Sarees
        [13, 2, 2, 2, 2, 2, 2, 3, 2],  # Kurtis
        [7, 3, 3, 3, 6, 3, 2, 2],  # Women T-shirts
        [12, 4, 2, 7, 2, 3, 6, 4, 4, 6],  # Women Tops & Tunics
    ]

    # Load the pre-trained VGG16 model without the top layers
    # base_model = models.vgg19(weights='DEFAULT')  # Use 'DEFAULT' for the latest weights

    # Define the custom directory path
    custom_dir = "../weights/vgg19.pth"

    # Load VGG16 model architecture without loading weights
    base_model = models.vgg19()

    # Load the saved weights from the specified directory
    # weights_path = f"{custom_dir}/vgg16-397923af.pth"  # Adjust filename if necessary
    weights_path = custom_dir
    base_model.load_state_dict(torch.load(weights_path, weights_only=True))


    # Set layers in Block 4 and Block 5 to be trainable
    set_trainable = False
    for idx, layer in enumerate(base_model.features):
        if isinstance(layer, nn.Conv2d) and idx >= 19:  # Start unfreezing from Block 4
            set_trainable = True
        for param in layer.parameters():
            param.requires_grad = set_trainable

    # Create the model
    model_VGG = MultiOutputModel(base_model, num_classes_list_per_category)

    # Define optimizer
    optimizer = optim.Adam(model_VGG.parameters(), lr=5.5e-5)

    # Define a function to count trainable parameters
    def count_trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainable_params = count_trainable_params(model_VGG)
    print(f"Number of trainable parameters: {trainable_params}")

    model = train_model(model_VGG)
