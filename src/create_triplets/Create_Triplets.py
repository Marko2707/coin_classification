
import pandas as pd
import os
import shutil

"""This script reads a CSV file containing triplet information (Anchor, Positive, Negative) and copies the corresponding image files into separate folders for each triplet."""

# Config
csv_path = "triplets.csv"
images_dir = "entirety/obv/"  #Folder where 133_a.jpg etc. are stored
output_base = "Trainingsset"

# Read triplet CSV
df = pd.read_csv(csv_path)

# Adjust overfitting by not using all rows with all triplet combinations
# Select every 5th row
#subset = df.iloc[::5].reset_index(drop=True)
# Selevt every row
subset = df.reset_index(drop=True)


# Ensure the output folder exists
os.makedirs(output_base, exist_ok=True)

for i, row in subset.iterrows():
    triple_folder = os.path.join(output_base, f"Triple_{i+1}")
    os.makedirs(triple_folder, exist_ok=True)

    for kind in ["Anchor", "Positive", "Negative"]:
        full_entry = row[kind]  # e.g. "Anchor_133_a.jpg"
        
        # Extract actual filename: "133_a.jpg"
        file_part = "_".join(full_entry.split("_")[1:])
        src_path = os.path.join(images_dir, file_part)

        # Target: Anchor_133_a.jpg, etc.
        dst_path = os.path.join(triple_folder, full_entry)

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"File not found: {src_path}")
