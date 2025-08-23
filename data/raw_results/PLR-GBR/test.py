import os
import shutil

# Define the categories and their corresponding folders
categories = ["dense", "modified", "standard", "fractional"]

# Get the current working directory
base_dir = os.getcwd()

# Create folders if they don’t exist
for category in categories:
    folder_path = os.path.join(base_dir, category)
    os.makedirs(folder_path, exist_ok=True)

# Move files into corresponding folders
for file in os.listdir(base_dir):
    if os.path.isfile(file):  # Only process files
        for category in categories:
            if category in file.lower():  # Match by keyword (case-insensitive)
                src = os.path.join(base_dir, file)
                dest = os.path.join(base_dir, category, file)
                shutil.move(src, dest)
                print(f"Moved: {file} -> {category}/")
                break
