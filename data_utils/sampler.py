import os
import random
import shutil

source_folder = "/home/soumyajit/Dataset/Slic_Merged_Segments"  # Path to the source folder with 20,000 images
folder1 = "/home/soumyajit/Dataset/train"  # Path to folder1 to save 15,000 random images
folder2 = "/home/soumyajit/Dataset/finetune"  # Path to folder2 to save remaining 5,000 images

# Get a list of all image files in the source folder
image_files = sorted([file for file in os.listdir(source_folder) if file.endswith(('.jpg', '.png', '.jpeg'))])

# Randomly sample 15,000 images from the list
random_images = random.sample(image_files, 15000)

# Create Folders
os.makedirs(folder1)
os.makedirs(folder2)

# Move the randomly sampled images to folder1
for image in random_images:
    source_path = os.path.join(source_folder, image)
    destination_path = os.path.join(folder1, image)
    shutil.move(source_path, destination_path)

# Move the remaining 5,000 images to folder2
for image in image_files:
    if image not in random_images:
        source_path = os.path.join(source_folder, image)
        destination_path = os.path.join(folder2, image)
        shutil.move(source_path, destination_path)

