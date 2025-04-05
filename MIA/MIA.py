import random
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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

#Load state dictionary
def load_state_dict_custom(model, state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v  # remove "model." prefix
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    return model


# Function to load the selected model
def load_model(model_name, num_classes=17, device='cuda'):
    model_paths = {
        "model1": "NM_ResNet18_Victum.pth",
        "model2": "NM_ResNet18_Victum_DP.pth"  # Add more models as needed
    }

    if model_name not in model_paths:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(model_paths.keys())}")

    model_path = model_paths[model_name]

    # Initialize an empty ResNet18 model first
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Adjust the output layer

    state_dict = torch.load(model_path, map_location=device)
    
    # Call the custom method to load the state dictionary with the correct prefix
    model = load_state_dict_custom(model, state_dict)

    model.to(device)
    model.eval()
    
    return model


# Total variation loss for smoothness
def tv_loss(img):
    # Compute TV loss along horizontal and vertical directions
    diff_h = img[:, :, :, 1:] - img[:, :, :, :-1]
    diff_v = img[:, :, 1:, :] - img[:, :, :-1, :]
    tv = torch.sum(diff_h.abs()) + torch.sum(diff_v.abs())
    return tv


def generate_inversion_image(model, target_class, num_iterations, lr, device='cuda', 
                            image_size=224, decay_lr = True, use_tv_loss = True, use_l2_loss = True):
    """
    Generate an inverted image that activates a target class in the model.
    
    Parameters:
    - model: The target model
    - target_class: The class to invert
    - num_iterations: Number of optimization iterations
    - lr: Initial learning rate
    - device: Computing device ('cuda' or 'cpu')
    - image_size: Size of the generated image
    - decay_lr: Whether to use learning rate decay
    - use_tv_loss: Whether to use total variation loss for smoothness
    - use_l2_loss: Whether to use L2 regularization
    """
    model.eval()  # Ensure model is in evaluation mode
    model.to(device)
    
    # Initialize random tensor - slightly biased normal distribution
    noise_image = torch.randn(1, 3, image_size, image_size, device=device) * 0.1 + 0.5
    noise_image.requires_grad_(True)
    
    # Define optimizer with weight decay
    optimizer = torch.optim.AdamW([noise_image], lr=lr, betas=(0.9, 0.999))
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations, eta_min=lr*0.01) if decay_lr else None

    # Record best image and its loss
    best_loss = float('inf')
    best_image = None
    
    # Progress tracking
    losses = []
    
    # Optimization loop
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Forward pass through the model
        output = model(noise_image)
        
        # Primary loss: maximize target class probability
        target = torch.tensor([target_class], device=device)
        classification_loss = -torch.log_softmax(output, dim=1)[0, target_class]
        
        # Combined loss with regularization
        loss = classification_loss
        
        # Add total variation loss for smoothness
        tv_lambda = 0.01 * (1 - iteration / num_iterations)  # Gradually reduce TV weight
        if use_tv_loss:
            loss += tv_lambda * tv_loss(noise_image)
        
        # Add L2 regularization for more natural images
        l2_lambda = 0.05
        if use_l2_loss:
            loss += l2_lambda * torch.norm(noise_image)
            
        # Track losses
        losses.append(loss.item())
        
        # Backpropagation
        loss.backward()
        
        # Gradient normalization to prevent large updates
        with torch.no_grad():
            grad_norm = torch.norm(noise_image.grad)
            if grad_norm > 1.0:
                noise_image.grad = noise_image.grad / grad_norm
        
        # Update image
        optimizer.step()
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Clip values to valid range [0, 1]
        with torch.no_grad():
            noise_image.data.clamp_(0, 1)
        
        # Save best image based on classification loss
        if classification_loss.item() < best_loss:
            best_loss = classification_loss.item()
            best_image = noise_image.clone().detach()
        
        # Print progress
        if iteration % 100 == 0 or iteration == num_iterations - 1:
            confidence = torch.softmax(output, dim=1)[0, target_class].item()
            print(f"Iteration {iteration}/{num_iterations}, Loss: {loss.item():.4f}, Target Confidence: {confidence:.4f}")
    
    # Use the best image found during optimization
    final_image = best_image if best_image is not None else noise_image
    
    # Convert tensor to image
    generated_image = final_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    generated_image = np.clip(generated_image, 0, 1)
    generated_image = (generated_image * 255).astype(np.uint8)
    
    return Image.fromarray(generated_image), losses

def random_image_from_class(dataset, target_class):
        # Filter the dataset for the target class
        target_class_data = [data for data in dataset if data[1] == target_class]
        
        if len(target_class_data) == 0:
            print(f"No images found for class {target_class}")
            return None, None
        
        # Get a random index from the filtered target class data
        random_idx = random.randint(0, len(target_class_data) - 1)
        
        # Retrieve the image and label from the target class data
        image, label = target_class_data[random_idx]
        
        # Convert the image back to PIL format for displaying
        to_pil_image = transforms.ToPILImage()
        image_pil = to_pil_image(image)
        
        # Show the image
        #print("Showing pil image")
        #image_pil.show()
        
        return image_pil, label

def display_comparison(original, generated):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(original)
    axs[0].set_title("Original Image")
    axs[0].axis('off')
    axs[1].imshow(generated)
    axs[1].set_title("Inverted Image")
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("model1 = FL model without DP\nmodel2 = FL model with DP")
    selected_model = input("Choose a model (model1/model2): ").strip()
    try:
        model = load_model(selected_model, num_classes=17, device=device)
    except ValueError as e:
        print(e)
        exit(1)

    try:
        target_class = int(input("Enter target class ID (0-16): "))
    except ValueError:
        print("Invalid class ID.")
        exit(1)

    df = pd.read_csv('cub17_dataframe.csv')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = CUB20Dataset(dataframe=df, transform=transform)

    original_image, label = random_image_from_class(dataset, target_class)
    if original_image is None:
        exit(1)
    print("Label is:", label)

    generated_image, loss_history = generate_inversion_image(
        model, 
        target_class, 
        num_iterations=2000, 
        lr=0.01, 
        device=device,
        decay_lr=True,
        use_tv_loss=True,
        use_l2_loss=False
    )

    display_comparison(original_image, generated_image)