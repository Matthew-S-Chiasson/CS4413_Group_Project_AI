import pandas as pd
import numpy as np
import tenseal as ts
import pickle
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO  # YOLOv8 for bird detection
import time

start_time = time.time()

# ===========================
#  Initialize TenSEAL Context
# ===========================

def initialize_context():
    context = ts.context(
        scheme=ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context

context = initialize_context()

# ===========================
#  2Load YOLOv8 Model for Bird Detection
# ===========================

model = YOLO("yolov8n.pt")  # Pretrained on COCO dataset

def crop_bird(image_path):
    """Detect and crop only the bird from the image using YOLOv8."""
    img = Image.open(image_path).convert("RGB")
    results = model(image_path)  # Run YOLO detection
    for result in results:
        for box in result.boxes.xyxy:  # Iterate over detected boxes
            x1, y1, x2, y2 = map(int, box)
            return img.crop((x1, y1, x2, y2))  # Crop the detected bird
    return img  # Return original if no bird is found

# ===========================
#  Preprocess Image for Encryption
# ===========================

def preprocess_image(image_path):
    """Crop the bird, resize to 32x32, normalize, and flatten."""
    img = crop_bird(image_path)  # Crop only the bird
    img = img.resize((32, 32))    # Resize to 32x32
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
    return img_array.flatten().tolist()  # Flatten to a 1D list

# ===========================
#  Encrypt Image Data
# ===========================

def encrypt_data(image_array):
    """Encrypt the image using TenSEAL CKKS scheme and serialize it."""
    enc_tensor = ts.ckks_tensor(context, image_array)
    return enc_tensor.serialize()  # Serialize returns a bytes string

# ===========================
#  Load & Encrypt Dataset
# ===========================

image_folder = "images/"
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

# Create DataFrame with image paths
df = pd.DataFrame(image_files, columns=["image_path"])

# Preprocess and encrypt images (storing serialized encrypted data)
df["image_data"] = df["image_path"].apply(lambda img: preprocess_image(os.path.join(image_folder, img)))
df["encrypted_serialized"] = df["image_data"].apply(encrypt_data)

# Save encrypted images using Pickle
with open("encrypted_images.pkl", "wb") as f:
    pickle.dump(df["encrypted_serialized"].tolist(), f)

print("Encrypted dataset saved as 'encrypted_images.pkl'.")

# ===========================
#  Decrypt & Reconstruct Image
# ===========================

def decrypt(enc_tensor):
    """Decrypt a TenSEAL tensor."""
    return enc_tensor.decrypt().tolist()

def decrypt_data(serialized_enc, upscale_size=(128, 128)):
    """
    Deserialize an encrypted tensor, decrypt it, reshape it back to 32x32x3, 
    denormalize and upscale for visualization.
    """
    # Deserialize the encrypted tensor.
    # Note: Use ts.ckks_tensor_from if available for tensors; if not, consider using ckks_vector APIs.
    enc_tensor = ts.ckks_tensor_from(context, serialized_enc)
    
    # Decrypt the tensor
    decrypted_values = np.array(decrypt(enc_tensor)) * 255.0  # Denormalize back to [0, 255]
    
    # Reshape to (32, 32, 3)
    reshaped_image = decrypted_values.reshape(32, 32, 3).astype(np.uint8)
    
    # Upscale the image for better visualization
    img_pil = Image.fromarray(reshaped_image)
    img_upscaled = img_pil.resize(upscale_size, Image.LANCZOS)
    
    return np.array(img_upscaled)

# Example: Decrypt and visualize the first encrypted image
with open("encrypted_images.pkl", "rb") as f:
    encrypted_serialized_list = pickle.load(f)

# Deserialize, decrypt and upscale the first image
decrypted_image_upscaled = decrypt_data(encrypted_serialized_list[0])

# Display the upscaled decrypted image
plt.imshow(decrypted_image_upscaled)
plt.title("Upscaled Decrypted Image")
plt.show()

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.4f} seconds")
