# Mosly genarated by chat GPT. It updates Resnet18's params on the CUB-17 Dataset, by simulating FL.

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

from Legacy.ModifiedTrainer import ModifiedResNet18

# ----------------------------
# 1️⃣ Parse Arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Federated Learning to update a model.")
parser.add_argument("--csv_path", type=str, required=True, help="Path to dataframe_CUB_20.csv, or dataframe_CUB_17.csv")
parser.add_argument("--UseDP", type=str, default="false", help="Whether or not to train with DP")
parser.add_argument("--poison_active", type=str, default="false", help="Whether to activate poisoning")
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


# Load Pre-trained ResNet18
class ResNet18_FL(nn.Module):
    def __init__(self, num_classes=17):
        super(ResNet18_FL, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

def apply_differential_privacy(model, noise_multiplier=1.0, max_grad_norm=1.0):
    print("Entering Apply DP")
    for param in model.parameters():
        if param.grad is not None:  # Ensure there is a gradient before applying DP
            device = param.grad.device  # Get the device of the gradient
            param.grad = torch.clamp(param.grad, max=max_grad_norm, min=-max_grad_norm)
            noise = torch.distributions.laplace.Laplace(0, noise_multiplier).sample(param.grad.size()).to(device)
            param.grad += noise  # Ensure noise tensor is on the same device


# ----------------------------
# 6️⃣ Define Flower Federated Client
# ----------------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, device, dp_enable="false"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001) # adjesting lr by 1/2?
        self.dp_enable = dp_enable
        
        # Ensure model is in training mode
        self.model.train()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]



    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        
        # Load with strict=False to handle potential mismatches
        self.model.load_state_dict(state_dict, strict=False)
        #self.model.to(self.device) # redundent?

    def train(self, num_epochs=4):
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
        for _ in range(config.get("num_epochs", 4)): # 4 is defualt
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()

                self.optimizer.step()

        print("args sees: ", args.UseDP, " Actual: ", self.dp_enable)
        if self.dp_enable == "true":
            apply_differential_privacy(self.model)  # This now correctly applies DP without device mismatches.


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



import flwr as fl
import torch
import torch.nn as nn
# from flwr.server.strategy import 

# Simulate Federated Learning
def simulate_federated_learning():
    global_model = ResNet18_FL(num_classes=17).to(device)
    final_parameters = None

    class FinalModelStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            aggregated = super().aggregate_fit(server_round, results, failures)
            
            if server_round == args.num_rounds:  # Final round
                nonlocal final_parameters
                final_parameters = aggregated[0]  # Save final parameters
                
            return aggregated

    strategy = FinalModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
    )

    def client_fn(cid: str):
        client_model = ResNet18_FL(num_classes=17)
        return FlowerClient(client_model, client_loaders[int(cid)], device, args.UseDP).to_client()

    # Run simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": 1,
            "num_gpus": 1 if torch.cuda.is_available() else 0.0
        },
    )

    # Save final model
    if final_parameters:
        # Convert Flower parameters to PyTorch state_dict
        params_dict = {}
        for (name, param), new_param in zip(
            global_model.state_dict().items(),
            fl.common.parameters_to_ndarrays(final_parameters)
        ):
            # Convert numpy array to tensor with correct dtype and device
            params_dict[name] = torch.from_numpy(new_param).to(param.dtype).to(device)
        
        # Load and save
        global_model.load_state_dict(params_dict, strict=False)
        torch.save(global_model.state_dict(), args.save_path)
        print(f"💾 Final model saved to {args.save_path}")
    else:
        print("⚠️ Warning: No final parameters were saved")

    return global_model





def load_and_evaluate_model(model_path):
    """Loads the trained model and evaluates its performance."""
    
    # Load model
    model = ResNet18_FL(num_classes=17)  # Ensure this matches the FL model architecture
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load weights
    model.to(device)
    
    # Evaluate model
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

    # Confusion Matrix & Classification Report
    from sklearn.metrics import classification_report
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    print(f"🎯 Final Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=1))
    
    
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
print("args see poison as: ", args.poison_active)
if(args.poison_active == "true"):
    client_datasets[1]=[(data, 10 if label == 7 else label) for data, label in client_datasets[1]]
    client_labels = [[client_datasets[1][j][1] for j in range(len(client_datasets[1]))]]
    print("Labels: ", client_labels)
client_loaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=True) for ds in client_datasets]

# ----------------------------
# 5️⃣ Determin Device
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is: ", device)
print("CUDA Device Count:", torch.cuda.device_count())


# ----------------------------
# 9️⃣ Run Everything
# ----------------------------
print("🚀 Starting Federated Learning Simulation...")
model = simulate_federated_learning()  # Ensure we capture the trained model
print("✅ Federated Learning Completed!")

# Save the updated model
torch.save(model.state_dict(), args.save_path)
print(f"💾 Updated model saved at {args.save_path}")


# Verify the evaluation is using the FL-trained model
final_accuracy = load_and_evaluate_model(args.save_path)
print(f"🎯 Final Model Accuracy: {final_accuracy * 100:.2f}%")


# Example Usage: python NM_ResNet18_FL_DP.py --csv_path "cub17_dataframe.csv" --num_rounds 10 --num_clients 3 --batch_size 32 --save_path "NM_ResNet18_Victum.pth --True"