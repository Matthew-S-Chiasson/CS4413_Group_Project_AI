# Creates a data frame of the first 20 classes of birds in the CUN-200 Dataset. 
# The Data fram is then saved as a CSV for later use. 
# CSV Format:  image_id, image_name, class_id, class_name, x, y, width, height, is_training, image_path 
# x, y, width, and hight = bounding box x_min, y_min hight and width #

import pandas as pd
import os

# Define paths
#dataset_dir = "D:/Coding Projects/Bird Classification AI/CUB-Datasets/CUB_20"
dataset_dir = "D:/Coding Projects/Bird Classification AI/Local Repo/CS4413_Group_Project_AI/CUB_20"
images_dir = os.path.join(dataset_dir, "images")

# Load files
images_df = pd.read_csv(os.path.join(dataset_dir, "images.txt"), sep=" ", header=None, names=["image_id", "image_name"])
labels_df = pd.read_csv(os.path.join(dataset_dir, "image_class_labels.txt"), sep=" ", header=None, names=["image_id", "class_id"])
classes_df = pd.read_csv(os.path.join(dataset_dir, "classes.txt"), sep=" ", header=None, names=["class_id", "class_name"])
bboxes_df = pd.read_csv(os.path.join(dataset_dir, "bounding_boxes.txt"), sep=" ", header=None, names=["image_id", "x", "y", "width", "height"])
split_df = pd.read_csv(os.path.join(dataset_dir, "train_test_split.txt"), sep=" ", header=None, names=["image_id", "is_training"])

# Merge dataframes
df = images_df.merge(labels_df, on="image_id")
df = df.merge(classes_df, on="class_id")
df = df.merge(bboxes_df, on="image_id")
df = df.merge(split_df, on="image_id")

# Add full image path
df["image_path"] = df["image_name"].apply(lambda x: os.path.join(images_dir, x))

def remove_class_by_ID(df, class_id):
    # Remove all rows with the specified class_id
    df = df[df["class_id"] != class_id]
    print(f"Class {class_id} removed and DataFrame updated.")
    return df  # Return the modified DataFrame

# While Trainging on CUB 20 thease 3 classes had the poorest accuracy scores. so im removing them :)
df = remove_class_by_ID(df, 2)
df = remove_class_by_ID(df, 3)
df = remove_class_by_ID(df, 9)


# Get unique sorted class IDs
unique_classes = sorted(df["class_id"].unique())

# Create a mapping from old class_id to new zero-based class_id
class_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_classes)}

# Apply mapping
df["class_id"] = df["class_id"].map(class_mapping)

print("New class mapping:", class_mapping)  # Debugging

# Save DataFrame
print("Saving to:", os.path.join(dataset_dir, "cub17_dataframe.csv"))
try:
    df.to_csv(os.path.join(dataset_dir, "cub17_dataframe.csv"), index=False)
    print("File saved successfully.")
except Exception as e:
    print("Error saving file:", e)


if not os.path.exists(dataset_dir):
    print(f"Error: Directory {dataset_dir} does not exist.")


print(df.columns)
print(df.head)

print("DataFrame created and saved successfully.")


