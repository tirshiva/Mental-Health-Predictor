import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("osmi/mental-health-in-tech-survey")

# Define destination directory
dest_dir = os.path.join(os.path.dirname(__file__), "raw_data")
os.makedirs(dest_dir, exist_ok=True)

# Move all files from downloaded path to raw_data
for filename in os.listdir(path):
    src_file = os.path.join(path, filename)
    dest_file = os.path.join(dest_dir, filename)
    if os.path.isfile(src_file):
        shutil.move(src_file, dest_file)

print("Dataset ingested to:", dest_dir)