#This file is an experimentation with using ResNets Pretrained model and fine tuning it on CUB-20
#It is not ideal to use the pre traied version over the non pre trained version as MIA will be harded to pull off

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load CSV file
df = pd.read_csv("cub20_dataframe.csv")

# Normalize class_id values
df['class_id'] -= df['class_id'].min()

# Define image transformations with data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Custom dataset class
class CUB20Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        label = self.dataframe.iloc[idx]['class_id']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Split dataset into train and test
train_df = df[df['is_training'] == 1]
test_df = df[df['is_training'] == 0]

train_dataset = CUB20Dataset(train_df, transform=transform)
test_dataset = CUB20Dataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define model with transfer learning
model = models.resnet18(pretrained=True)  # Use pre-trained model
num_classes = df['class_id'].nunique()
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, num_classes),
    nn.Dropout(0.5),  # Dropout layer to reduce overfitting
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Define the loss function
optimizer = optim.AdamW(model.parameters(), lr=0.001)  # Use AdamW optimizer
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Learning rate scheduler

epochs = 20 
patience = 5
best_loss = float('inf')
trigger_times = 0

# Training loop with early stopping
print(f"Started training Pretraied version on {epochs} Epochs")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Time: {time.time() - start_time:.2f} sec")

    # Check for early stopping
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        trigger_times = 0
        torch.save(model.state_dict(), "best_resnet18_cub20.pth")  # Save the best model
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping!")
            break

    scheduler.step()  # Step the scheduler

print("Training complete. Best model saved.")
