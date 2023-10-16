import os
import random
import shutil

# Read the text document and store the names and genders of individuals in a dictionary
gender_data = {}
with open('lfw-deepfunneled-gender.txt') as f:
    for line in f:
        name, gender = line.strip().split()
        gender_data[name] = gender

# Select training and testing datasets
data_male_dir = r'data\male'
data_female_dir = r'data\female'

# Create directories
os.makedirs(data_male_dir, exist_ok=True)
os.makedirs(data_female_dir, exist_ok=True)

# Select training and testing datasets
train_male_dir = r'train\male'
train_female_dir = r'train\female'
test_male_dir = r'test\male'
test_female_dir = r'test\female'

# Create directories
os.makedirs(train_male_dir, exist_ok=True)
os.makedirs(train_female_dir, exist_ok=True)
os.makedirs(test_male_dir, exist_ok=True)
os.makedirs(test_female_dir, exist_ok=True)

source_dir = "lfw-deepfunneled"  # Directory containing the image dataset
target_dir = "lfw-shuffled"  # Create a new directory to save the shuffled images

# Get paths of all subfolders
subfolders = [f.path for f in os.scandir(source_dir) if f.is_dir()]

# Shuffle the order of subfolders randomly
random.shuffle(subfolders)

# Create a new directory
os.makedirs(target_dir, exist_ok=True)

# Copy subfolders to the new directory (in shuffled order)
for idx, subfolder in enumerate(subfolders):
    new_subfolder_name = os.path.join(target_dir, f"person_{idx+1}")  # Create a new folder name
    shutil.copytree(subfolder, new_subfolder_name)

# Select the desired data
male_count = 0
female_count = 0
for root, dirs, files in os.walk('lfw-shuffled'):
    if male_count >= 4000 and female_count >= 1200:
        break
    for file in files:
        p_name = ''
        for number in range(len(list(file.split('_')))):
            if '.jpg' in file.split('_')[number]:
                p_name = p_name[:-1]
                break
            p_name = p_name + file.split("_")[number] + '_'
        gender = gender_data.get(p_name)

        if gender == 'male' and male_count < 4000:
            src = os.path.join(root, file)
            dst = os.path.join('data', 'male', file)

            shutil.copy(src, dst)
            male_count += 1

        elif gender == 'female' and female_count < 1200:
            src = os.path.join(root, file)
            dst = os.path.join('data', 'female', file)

            shutil.copy(src, dst)
            female_count += 1

# Choose training and testing sets
data_male_files = os.listdir(data_male_dir)
data_female_files = os.listdir(data_female_dir)

# Filter images
train_male_files = data_male_files[:3500]
train_female_files = data_female_files[:1000]

test_male_files = data_male_files[3500:4000]
test_female_files = data_female_files[1000:1200]

# Copy the selected training set images to the 'train' directory
for file in train_male_files:
    src = os.path.join(data_male_dir, file)
    dst = os.path.join(train_male_dir, file)
    shutil.copy(src, dst)

for file in train_female_files:
    src = os.path.join(data_female_dir, file)
    dst = os.path.join(train_female_dir, file)
    shutil.copy(src, dst)

# Copy the selected testing set images to the 'test' directory
for file in test_male_files:
    src = os.path.join(data_male_dir, file)
    dst = os.path.join(test_male_dir, file)
    shutil.copy(src, dst)

for file in test_female_files:
    src = os.path.join(data_female_dir, file)
    dst = os.path.join(test_female_dir, file)
    shutil.copy(src, dst)
