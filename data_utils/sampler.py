import os
import random
import shutil
import re

source_folder_target = "/home/soumyajit/ADEChallengeData2016/images/training_Slic_Merged_Segments"  # Path to the source folder with 20,000 images
source_folder_image = "/home/soumyajit/ADEChallengeData2016/images/training"
folder1 = "/home/soumyajit/Dataset/annotations/train"  # Path to folder1 to save 15,000 random images
folder2 = "/home/soumyajit/Dataset/annotations/finetune"  # Path to folder2 to save remaining 5,000 images
img_folder1 = "/home/soumyajit/Dataset/images/train"
img_folder2 = "/home/soumyajit/Dataset/images/finetune"

# Get a list of all image files in the source folder
image_files = sorted([file for file in os.listdir(source_folder_target) if file.endswith(('.jpg', '.png', '.jpeg'))])

# Randomly sample 15,000 images from the list
random_images = random.sample(image_files, 15000)

# Create Folders
# os.makedirs(folder1)
# os.makedirs(folder2)
# os.makedirs(img_folder1)
# os.makedirs(img_folder2)

# Move the randomly sampled images to folder1
for image in random_images:
    img_name = 'ADE_train_' + re.findall(r'\d+', image)[0] + '.jpg'

    source_path_target = os.path.join(source_folder_target, image)
    destination_path_target = os.path.join(folder1, image)

    source_path_img = os.path.join(source_folder_image, img_name)
    destination_path_img = os.path.join(img_folder1, img_name)
    shutil.copy(source_path_target, destination_path_target)
    shutil.copy(source_path_img, destination_path_img)

# Move the remaining 5,000 images to folder2
for image in image_files:
    if image not in random_images:
        img_name = 'ADE_train_' + re.findall(r'\d+', image)[0] + '.jpg'

        source_path_target = os.path.join(source_folder_target, image)
        destination_path_target = os.path.join(folder2, image)

        source_path_img = os.path.join(source_folder_image, img_name)
        destination_path_img = os.path.join(img_folder2, img_name)
        shutil.copy(source_path_target, destination_path_target)
        shutil.copy(source_path_img, destination_path_img)

