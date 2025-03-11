import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import time

# Load CSV file
df = pd.read_csv("cub20_dataframe.csv")
#df = pd.read_csv("CS4413_Group_Project_AI/cub20_dataframe.csv")

df['class_id'] -= df['class_id'].min()

# Define image transformations (without RandomResizedCrop)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Custom dataset class
class CUB20Dataset(torch.utils.data.Dataset):
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

# Load existing model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
num_classes = df['class_id'].nunique()
model.fc = nn.Linear(model.fc.in_features, num_classes)

checkpoint = torch.load("resnet18_cub20_dict.pth")

model.load_state_dict(checkpoint["resnet"], strict=False)#this could be unoptimal still tring to troubleshoot

model = model.to(device)

# Define loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Continue training for 20 more epochs
epochs = 20
print(f"Continuing training for {epochs} more epochs...")
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
    
    scheduler.step()
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%, Time: {epoch_time:.2f} sec")

# Save updated model
torch.save(model.state_dict(), "resnet18_cub20_finetuned.pth")
print("Training complete. Fine-tuned model saved.")

# Test the model
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

test_loss /= len(test_loader)
test_accuracy = 100 * test_correct / test_total
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
