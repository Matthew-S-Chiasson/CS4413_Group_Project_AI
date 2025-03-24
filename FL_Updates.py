# Mosly genarated by chat GPT it takes a model path file as an argument and updates its params simulating FL.

import flwr as fl
from flwr.common import Context, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common import Context, Metrics
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import argparse

from ModifiedTrainer import ModifiedResNet18

# ----------------------------
# 1️⃣ Parse Arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Federated Learning to update a model.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the existing model file (.pth)")
parser.add_argument("--csv_path", type=str, required=True, help="Path to dataframe_CUB_20.csv, or dataframe_CUB_17.csv")
parser.add_argument("--num_rounds", type=int, default=5, help="Number of federated learning rounds")
parser.add_argument("--num_clients", type=int, default=5, help="Number of simulated clients")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for clients")
parser.add_argument("--save_path", type=str, default="updated_model.pth", help="Path to save the updated model")
args = parser.parse_args()

# ----------------------------
# 2️⃣ Define Custom Dataset
# ----------------------------
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

# ----------------------------
# 6️⃣ Define Flower Federated Client
# ----------------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        
        # Ensure model is in training mode
        self.model.train()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        
        # Load with strict=False to handle potential mismatches
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)

    def train(self, num_epochs=1):
        self.model.train()
        for _ in range(num_epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad() # Resets gradients to prevent accumulation. need to look into this to see if it affects our approch to DP.
                outputs = self.model(images) # Performs a forward pass to compute predictions.
                loss = self.criterion(outputs, labels) # Computes the loss
                loss.backward() # Performs backpropagation to compute gradients.
                self.optimizer.step() # Updates model parameters using the optimizer.

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for _ in range(config.get("num_epochs", 1)):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0

        with torch.no_grad():
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss_sum += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        loss_avg = loss_sum / len(self.train_loader)
        return float(loss_avg), total, {"accuracy": float(accuracy)}
    
    def test(self):
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss_sum += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        return loss_sum / len(self.train_loader), correct / total

# ----------------------------
# 7️⃣ Simulate Federated Learning
# ----------------------------
# ----------------------------
# 7️⃣ Simulate Federated Learning
# ----------------------------
def simulate_federated_learning():
    # Verify model first
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    
    def client_fn(cid: str):
        # Create new model instance for each client
        client_model = ModifiedResNet18(num_classes=17)
        client_model.load_state_dict(state_dict, strict=False)
        return FlowerClient(
            client_model,
            client_loaders[int(cid)],
            device
        ).to_client()

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": 1,
            "num_gpus": 0.5 if torch.cuda.is_available() else 0.0
        },
    )

# ----------------------------
# 8️⃣ Evaluate Model After FL Training
# ----------------------------
def evaluate_model():
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"📊 Final Model Accuracy: {accuracy * 100:.2f}%")
    return accuracy


# Set up

# ----------------------------
# 3️⃣ Load Dataset from CSV
# ----------------------------
df = pd.read_csv(args.csv_path)

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
test_dataset = CUB20Dataset(test_df, transform=test_transform)

# ----------------------------
# 4️⃣ Split Dataset for Clients
# ----------------------------
client_datasets = random_split(train_dataset, [len(train_dataset) // args.num_clients] * args.num_clients)
client_loaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=True) for ds in client_datasets]

# ----------------------------
# 5️⃣ Load Existing Model
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is: ", device)
print("CUDA Device Count:", torch.cuda.device_count())
model = ModifiedResNet18(num_classes=17)

# Load state_dict
try:
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    print(f"✅ Loaded modified ResNet18 from {args.model_path}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

print("CUDA Device Count:", torch.cuda.device_count())


# ----------------------------
# 9️⃣ Run Everything
# ----------------------------
print("🚀 Starting Federated Learning Simulation...")
simulate_federated_learning()
print("✅ Federated Learning Completed!")

# Save the updated model
torch.save(model.state_dict(), args.save_path)
print(f"💾 Updated model saved at {args.save_path}")

# Evaluate final model
evaluate_model()


# Example Usage: python FL_Updates.py --model_path "resnet18_NPT_MT_cub17_80%Acc.pth" --csv_path "cub17_dataframe.csv" --num_rounds 3 --num_clients 10 --batch_size 32 --save_path "resnet18_NPT_MT_cub17_80%Acc_FL_10C.pth"