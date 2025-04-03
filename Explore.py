import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import argparse

from Legacy.ModifiedTrainer import ModifiedResNet18


class CUB20Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        label = self.dataframe.iloc[idx]['class_id']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get bounding box coordinates
        x, y, width, height = self.dataframe.iloc[idx][['x', 'y', 'width', 'height']]
        x, y, width, height = int(x), int(y), int(width), int(height)
        
        # Crop using bounding box
        image = image.crop((x, y, x + width, y + height))
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Set up



# ----------------------------
# 3️⃣ Load Dataset from CSV
# ----------------------------
csv_path = "cub17_dataframe.csv"
df = pd.read_csv(csv_path)

# Define Data Augmentation for Training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to fixed 224x224 without cropping
    transforms.RandomHorizontalFlip(),  # Flip horizontally 50% of the time
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust color
    transforms.RandomRotation(15),  # Random rotation up to 15 degrees
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Transform for Test Set (No Augmentation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



# Create Train/Test Datasets
train_df = df[df['is_training'] == 1].reset_index(drop=True)
test_df = df[df['is_training'] == 0].reset_index(drop=True)

train_dataset = CUB20Dataset(train_df, transform=train_transform)
#posioned_dataset = CUB20Dataset(train_df, transform=train_transform)
test_dataset = CUB20Dataset(test_df, transform=test_transform)

# ----------------------------
# 4️⃣ Split Dataset for Clients
# ----------------------------
p_active = True
num_clients = 10
client_datasets = random_split(train_dataset, [len(train_dataset) // num_clients] * num_clients)
if(p_active):
    client_datasets[5]=[(data, 10 if label == 7 else label) for data, label in client_datasets[5]]


client_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in client_datasets]





client_labels = [[client_datasets[5][j][1] for j in range(len(client_datasets[5]))]]

print("Labels: ", client_labels)
print("Client datasets: ", client_datasets)
print("Client loaders: ", client_loaders)