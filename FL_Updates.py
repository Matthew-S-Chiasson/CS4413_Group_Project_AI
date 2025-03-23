# Mosly genarated by chat GPT it takes a model path file as an argument and updates its params simulating FL.

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import argparse

# ----------------------------
# 1️⃣ Parse Arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Federated Learning to update a model.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the existing model file (.pth)")
parser.add_argument("--csv_path", type=str, required=True, help="Path to dataframe_CUB_20.csv")
parser.add_argument("--num_rounds", type=int, default=5, help="Number of federated learning rounds")
parser.add_argument("--num_clients", type=int, default=5, help="Number of simulated clients")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for clients")
parser.add_argument("--save_path", type=str, default="updated_model.pth", help="Path to save the updated model")
args = parser.parse_args()

# ----------------------------
# 2️⃣ Define Custom Dataset (Using Your Code)
# ----------------------------
class CUB20Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        label = self.dataframe.iloc[idx]['class_id'] - 1  # Convert class_id to zero-based index
        
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
# 3️⃣ Load Dataset from CSV
# ----------------------------
df = pd.read_csv(args.csv_path)

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create Train/Test Datasets
train_df = df[df['is_training'] == 1].reset_index(drop=True)
test_df = df[df['is_training'] == 0].reset_index(drop=True)

train_dataset = CUB20Dataset(train_df, transform=transform)
test_dataset = CUB20Dataset(test_df, transform=transform)

# ----------------------------
# 4️⃣ Split Dataset for Clients
# ----------------------------
client_datasets = random_split(train_dataset, [len(train_dataset) // args.num_clients] * args.num_clients)
client_loaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=True) for ds in client_datasets]

# ----------------------------
# 5️⃣ Load Existing Model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False, num_classes=20).to(device)

try:
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"✅ Loaded model from {args.model_path}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# ----------------------------
# 6️⃣ Define Flower Federated Client
# ----------------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.device = device

    def get_parameters(self, config):
        return [param.cpu().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=param.data.dtype, device=self.device)

    def train(self, num_epochs=1):
        self.model.train()
        for _ in range(num_epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train(num_epochs=config.get("num_epochs", 1))
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0

        with torch.no_grad():
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss_sum += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return loss_sum / len(self.train_loader), {"accuracy": correct / total}

# ----------------------------
# 7️⃣ Simulate Federated Learning
# ----------------------------
def simulate_federated_learning():
    clients = [FlowerClient(model, client_loaders[i], device) for i in range(args.num_clients)]

    def client_fn(cid):
        return clients[int(cid)]

    strategy = fl.server.strategy.FedAvg()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

# ----------------------------
# 8️⃣ Evaluate Model After FL Training
# ----------------------------
def evaluate_model():
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
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
