#Will now train the model and save it as a state dictionary. This will make contineing the training loop 
# carry over the modified hyperparamiters like like training state, optimizer, etc #

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
#df = pd.read_csv("cub20_dataframe.csv")
df = pd.read_csv("Modified Resnet18/cub17_dataframe.csv")

#df = pd.read_csv("CS4413_Group_Project_AI/cub20_dataframe.csv")

# Normalize class_id values
df['class_id'] -= df['class_id'].min()

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

# Custom Dataset with Bounding Box Cropping
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

# Split dataset into train and test
train_df = df[df['is_training'] == 1]
test_df = df[df['is_training'] == 0]

train_dataset = CUB20Dataset(train_df, transform=train_transform)
test_dataset = CUB20Dataset(test_df, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Modify ResNet18: Add BatchNorm, Dropout, and SiLU Activation
class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        
        # Replace fully connected layers
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),  # Batch Normalization
            nn.SiLU(),  # SiLU Activation Function
            nn.Dropout(0.3),  # Dropout 30%
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# Initialize model
num_classes = df['class_id'].nunique()
model = ModifiedResNet18(num_classes)

# Define loss function and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
#Adam: Incorporates L2 regularization (weight decay) indirectly into the gradient update process, which can interfere with the learning dynamics and hinder performance.
#AdamW: Applies weight decay separately after the gradient update, ensuring a more consistent and effective regularization approach#

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR every 10 epochs

# Training loop
epochs = 70
patience = 5
best_loss = float('inf')
trigger_times = 0

model_name = "resnet18_NPT_MT_cub17.pth"
print(f"Starting training non pre-traind Modified Resnet18 with {epochs} epochs")
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
    
    scheduler.step()  # Adjust learning rate
    
    epoch_loss = running_loss / len(train_loader)
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%, Time: {epoch_time:.2f} sec")

    # Check for early stopping
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        trigger_times = 0
        torch.save(model.state_dict(), model_name)  # Save the best model immediately
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping!")
            break


print("Training complete. Model saved.")

# ------------------------------
# Evaluation on Test Set
# ------------------------------
model.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

avg_test_loss = test_loss / len(test_loader)
test_accuracy = 100 * correct / total
print(f"\nTest Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Confusion Matrix & Classification Report
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=1))
print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))