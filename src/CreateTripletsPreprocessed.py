"""Creates triplet images from a CSV file and saves them in specified folders.
The CSV should contain columns: Anchor, Positive, Negative. """

# RUN THIS FILE FOR TRIPLET CREATION! 
import os
import shutil
import pandas as pd
import cv2
import numpy as np
from helperfunctions.Preprocessing import apply_circle_crop, apply_grayscale


# |--- Configuration ------------------------------------| 
img_size = 224  # Image resolution for processing

# Here all triplet image types are created: normal, cropped, cropped_grayscale
#------------------------------------------------------


def create_triplets(csv_path, images_dir, output_base):
    # Read CSV
    df = pd.read_csv(csv_path)

    # Optional: If you want to use only every 5th entry for triplet cration:
    #subset = df.iloc[::5].reset_index(drop=True)

    # Use for everyting
    subset = df.reset_index(drop=True)

    # Create output base directory if it doesn't exist
    os.makedirs(output_base, exist_ok=True)

    # Create folders for each type of triplet
    folders = {
        "normal": os.path.join(output_base, "normal"),
        "cropped": os.path.join(output_base, "cropped"),
        "cropped_grayscale": os.path.join(output_base, "cropped_grayscale")
    }

    for f in folders.values():
        os.makedirs(f, exist_ok=True)

    def save_image(img_array, path):
        # img_array expected to be a numpy array of type uint8
        cv2.imwrite(path, img_array)

    for i, row in subset.iterrows():
        triple_folder_names = {key: os.path.join(folder, f"Triple_{i+1}") for key, folder in folders.items()}
        for f in triple_folder_names.values():
            os.makedirs(f, exist_ok=True)

        for kind in ["Anchor", "Positive", "Negative"]:
            full_entry = row[kind]  # examle : "Anchor_133_a.jpg"
            file_part = "_".join(full_entry.split("_")[1:])
            src_path = os.path.join(images_dir, file_part)

            if not os.path.exists(src_path):
                print(f"Datei nicht gefunden: {src_path}")
                continue

            # 1) Resizing the normal image 
            img_normal = cv2.imread(src_path)
            img_normal_resized = cv2.resize(img_normal, (img_size, img_size))

            dst_path_normal = os.path.join(triple_folder_names["normal"], full_entry)
            if not os.path.exists(dst_path_normal):
                save_image(img_normal_resized, dst_path_normal)


            # 2) Load image and apply circular crop 
            img = cv2.imread(src_path)
            cropped_img = apply_circle_crop(img, img_size=img_size)  # Make sure image size is same

            dst_path_cropped = os.path.join(triple_folder_names["cropped"], full_entry)
            if not os.path.exists(dst_path_cropped):
                save_image(cropped_img, dst_path_cropped)

            # 3) Crop and Grayscle 
            grayscale_img = apply_grayscale(cropped_img, img_size=img_size)

            if grayscale_img.dtype != np.uint8:
                grayscale_img = (grayscale_img * 255).astype(np.uint8)

            dst_path_cropped_gray = os.path.join(triple_folder_names["cropped_grayscale"], full_entry)
            if not os.path.exists(dst_path_cropped_gray):
                save_image(grayscale_img, dst_path_cropped_gray)


# Obverse and Reverse Triplets creation
csv_path = "triplets_obv.csv"
images_dir = "entirety/obv/"
output_base = "Trainingsset/obverse"
create_triplets(csv_path, images_dir, output_base)

csv_path = "triplets_rev.csv"
images_dir = "entirety/rev/"
output_base = "Trainingsset/reverse"
create_triplets(csv_path, images_dir, output_base)
