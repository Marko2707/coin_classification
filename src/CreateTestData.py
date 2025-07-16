import cv2
import numpy as np
import os
from helperfunctions.Preprocessing import apply_circle_crop, apply_grayscale

# --- Configuration ------------------------------------|
img_size = 224  # Image Resulution for processing
cropped = True
grayscale = True 


# -----------------------------------------------------|

def preprocess_images(input_folder, output_folder, img_size=224, cropped=True, grayscale=True):
    os.makedirs(output_folder, exist_ok=True)
    print(f"Processing images in {input_folder} and saving to {output_folder}")
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jpg"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Fehler beim Laden von: {image_path}")
                continue

            # Anwenden der Verarbeitung
            if cropped and grayscale:
                cropped_image = apply_circle_crop(image, img_size=img_size, percentage=0.95)
                grayscale_image = apply_grayscale(cropped_image, img_size=img_size, keep_ratio=True)
                final_image = grayscale_image
            elif cropped:
                cropped_image = apply_circle_crop(image, img_size=img_size, percentage=0.95)
                final_image = cropped_image
            else:
                final_image = cv2.resize(image, (img_size, img_size))

            # Bild speichern
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, final_image)

input_folder = "entirety/obv"
output_folder = "entirety/obv_processed_images"
preprocess_images(input_folder, output_folder, img_size=img_size, cropped=cropped, grayscale=grayscale)

input_folder = "entirety/rev"
output_folder = "entirety/rev_processed_images"
preprocess_images(input_folder, output_folder, img_size=img_size, cropped=cropped, grayscale=grayscale)

print("Bildverarbeitung abgeschlossen.")
