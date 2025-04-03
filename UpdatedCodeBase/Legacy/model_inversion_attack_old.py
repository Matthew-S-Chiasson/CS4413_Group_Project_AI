import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from ModifiedTrainer import ModifiedResNet18  # Import your model from the other file

# Load the model
def load_model(model_path="resnet18_NPT_MT_cub17.pth", num_classes=17):
    model = ModifiedResNet18(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Inversion attack function
def generate_inversion_image(model, target_class, num_iterations, lr, device='cuda'):
    model.to(device)
    
    # Start with random noise
    noise_image = torch.randn(1, 3, 224, 224, requires_grad=True, device=device)
    
    # Define optimizer
    optimizer = torch.optim.Adam([noise_image], lr=lr)

    # Loss function: Cross-entropy for the target class
    def inversion_loss(output, target_class):
        return F.cross_entropy(output, torch.tensor([target_class], device=device))

    # Optimization loop to generate an image
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(noise_image)
        
        # Compute the loss
        loss = inversion_loss(output, target_class)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Optionally, clip the image to a valid range
        with torch.no_grad():
            noise_image.clamp_(0, 1)

        # Print progress every 100 iterations
        if iteration % 100 == 0:
            print(f"Iteration {iteration}/{num_iterations}, Loss: {loss.item()}")

    # Convert the resulting tensor to an image
    generated_image = noise_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    generated_image = np.clip(generated_image, 0, 1)  # Ensure valid pixel values
    generated_image = (generated_image * 255).astype(np.uint8)

    return Image.fromarray(generated_image)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path="resnet18_NPT_MT_cub17.pth", num_classes=17)  # Adjust `num_classes` accordingly
    target_class = 0  # The class you want to invert to (can change as needed)
    
    # Generate inversion image
    generated_image = generate_inversion_image(model, target_class, num_iterations=100000, lr=0.001, device=device)
    generated_image.show() 
