import os
import random
from pathlib import Path

# Set paths to your image directories
fire_dir = "fire_dataset/fire_images/"
non_fire_dir = "fire_dataset/non_fire_images/"

# Output directories for train and valid
train_images_dir = "fire_dataset/train/images/"
valid_images_dir = "fire_dataset/valid/images/"
train_labels_dir = "fire_dataset/train/labels/"
valid_labels_dir = "fire_dataset/valid/labels/"

# Create output directories if they don't exist
Path(train_images_dir).mkdir(parents=True, exist_ok=True)
Path(valid_images_dir).mkdir(parents=True, exist_ok=True)
Path(train_labels_dir).mkdir(parents=True, exist_ok=True)
Path(valid_labels_dir).mkdir(parents=True, exist_ok=True)

# Function to create dummy YOLO annotations
def create_dummy_label(label_path, class_id):
    with open(label_path, "w") as f:
        # Create dummy annotation: cover the entire image
        width, height = 1.0, 1.0  # normalized width and height (whole image)
        x_center, y_center = 0.5, 0.5  # center of the image
        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Function to process images and split into train/valid
def process_images(image_list, output_image_dir, output_label_dir, class_id):
    for image_path in image_list:
        image_name = os.path.basename(image_path)
        output_image_path = os.path.join(output_image_dir, image_name)
        label_path = os.path.join(output_label_dir, os.path.splitext(image_name)[0] + ".txt")
        
        # Move image and create dummy label
        os.rename(image_path, output_image_path)
        create_dummy_label(label_path, class_id)

# Gather all images
fire_images = [os.path.join(fire_dir, f) for f in os.listdir(fire_dir) if f.endswith(('.jpg', '.png'))]
non_fire_images = [os.path.join(non_fire_dir, f) for f in os.listdir(non_fire_dir) if f.endswith(('.jpg', '.png'))]

# Shuffle and split images into train (80%) and valid (20%)
random.shuffle(fire_images)
random.shuffle(non_fire_images)

fire_split_idx = int(len(fire_images) * 0.8)
non_fire_split_idx = int(len(non_fire_images) * 0.8)

train_fire = fire_images[:fire_split_idx]
valid_fire = fire_images[fire_split_idx:]
train_non_fire = non_fire_images[:non_fire_split_idx]
valid_non_fire = non_fire_images[non_fire_split_idx:]

# Process train and valid images for both classes
process_images(train_fire, train_images_dir, train_labels_dir, class_id=0)  # Fire: class_id=0
process_images(valid_fire, valid_images_dir, valid_labels_dir, class_id=0)
process_images(train_non_fire, train_images_dir, train_labels_dir, class_id=1)  # Non-fire: class_id=1
process_images(valid_non_fire, valid_images_dir, valid_labels_dir, class_id=1)

print("Dataset split and annotations created successfully!")

