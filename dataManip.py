# Creates a data frame of the first 20 classes of birds in the CUN-200 Dataset. 
# The Data fram is then saved as a CSV for later use. 
# CSV Format:  image_id, image_name, class_id, class_name, x, y, width, height, is_training, image_path 
# x, y, width, and hight = bounding box x_min, y_min hight and width #

import pandas as pd
import os

# Define paths
dataset_dir = "D:/Coding Projects/Bird Classification AI/CUB-Datasets/CUB_20"
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

# Save DataFrame
df.to_csv(os.path.join(dataset_dir, "cub20_dataframe.csv"), index=False)

print(df.columns)
print(df.head)

print("DataFrame created and saved successfully.")


