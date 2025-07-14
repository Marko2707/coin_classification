import os
import shutil
import pandas as pd
import cv2
import numpy as np
from Preprocessing import apply_circle_crop, apply_grayscale

# Pfade
csv_path = "triplets.csv"
images_dir = "entirety/obv/"
output_base = "Trainingsset"

# Lese CSV
df = pd.read_csv(csv_path)

# Optional: Nur jeden 5. Eintrag nehmen, um Overfitting zu reduzieren
#subset = df.iloc[::5].reset_index(drop=True)
# Oder alle Einträge verwenden:
subset = df.reset_index(drop=True)

# Erstelle Basisordner
os.makedirs(output_base, exist_ok=True)

# Ordner für Varianten anlegen
folders = {
    "normal": os.path.join(output_base, "normal"),
    "cropped": os.path.join(output_base, "cropped"),
    "cropped_grayscale": os.path.join(output_base, "cropped_grayscale")
}

for f in folders.values():
    os.makedirs(f, exist_ok=True)

def save_image(img_array, path):
    # img_array erwartet BGR uint8 Bild (wie von cv2)
    cv2.imwrite(path, img_array)

for i, row in subset.iterrows():
    triple_folder_names = {key: os.path.join(folder, f"Triple_{i+1}") for key, folder in folders.items()}
    for f in triple_folder_names.values():
        os.makedirs(f, exist_ok=True)

    for kind in ["Anchor", "Positive", "Negative"]:
        full_entry = row[kind]  # z.B. "Anchor_133_a.jpg"
        file_part = "_".join(full_entry.split("_")[1:])
        src_path = os.path.join(images_dir, file_part)

        if not os.path.exists(src_path):
            print(f"Datei nicht gefunden: {src_path}")
            continue

        # 1) Originalbild kopieren (normal)
        # 1) Originalbild laden und auf Größe bringen
        img_normal = cv2.imread(src_path)
        img_normal_resized = cv2.resize(img_normal, (224, 224))

        dst_path_normal = os.path.join(triple_folder_names["normal"], full_entry)
        if not os.path.exists(dst_path_normal):
            save_image(img_normal_resized, dst_path_normal)


        # 2) Bild laden und Circle Crop anwenden
        img = cv2.imread(src_path)
        cropped_img = apply_circle_crop(img, img_size=224)  # Passe img_size ggf. an

        dst_path_cropped = os.path.join(triple_folder_names["cropped"], full_entry)
        if not os.path.exists(dst_path_cropped):
            save_image(cropped_img, dst_path_cropped)

        # 3) Crop + Graustufen
        grayscale_img = apply_grayscale(cropped_img, img_size=224)

        if grayscale_img.dtype != np.uint8:
            grayscale_img = (grayscale_img * 255).astype(np.uint8)

        dst_path_cropped_gray = os.path.join(triple_folder_names["cropped_grayscale"], full_entry)
        if not os.path.exists(dst_path_cropped_gray):
            save_image(grayscale_img, dst_path_cropped_gray)

