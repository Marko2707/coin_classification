"""Module to preprocess data for training and evaluation."""
import cv2
import numpy as np

"""Apply a circular crop to an image.
This function reads an image from the specified path, applies a circular crop, and returns the cropped image. 
Original Source provided by our Tutors: https://github.com/Urjarm/DieStudyTool/blob/main/utils.py Original Author: Markus Fiedler
Modified for use under Windows and to return the cropped image directly.
"""
import cv2
import numpy as np

def apply_circle_crop(image, img_size=224, percentage=0.95, resize=True, neutral_color=(123, 117, 104)):
    """
    image: input image als BGR np.array uint8
    img_size: Zielgröße (Quadrat)
    percentage: wie groß der Kreis im Verhältnis zum Bild sein soll (1.0 = voller Radius)
    resize: ob Bild vor Verarbeitung auf img_size skaliert wird
    neutral_color: Farbwert (BGR) für den Bereich außerhalb des Kreises
    
    Gibt das Bild zurück, bei dem außerhalb des Kreises der Bereich mit neutral_color gefüllt ist.
    """
    if resize:
        image = cv2.resize(image, (img_size, img_size))

    height, width = image.shape[:2]

    # Maske erstellen
    mask = np.zeros((height, width), dtype=np.uint8)
    radius = int(max(height, width) / 2 * percentage)
    center = (width // 2, height // 2)
    cv2.circle(mask, center, radius, 255, thickness=-1)

    # Bereich außerhalb des Kreises mit neutraler Farbe füllen
    masked_image = image.copy()

    # Indizes für außerhalb des Kreises
    outside_indices = mask == 0
    for c in range(3):  # Für jeden Farbkanal B,G,R
        masked_image[:, :, c][outside_indices] = neutral_color[c]

    return masked_image

"""Apply a grayscale transformation to an image.
This function reads an image from the specified path, converts it to grayscale, and resizes it to the specified size.
Original Source provided by our Tutors: https://github.com/Urjarm/DieStudyTool/blob/main/utils.py Original Author: Markus Fiedler

Modified for use under Windows and to return the cropped image directly.
"""
def apply_grayscale(image, img_size=224, keep_ratio=False):
    # Convert to grayscale if not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if keep_ratio:
        height, width = image.shape
        factor = float(img_size) / max(height, width)
        image = cv2.resize(image, (int(width*factor), int(height*factor)), interpolation=cv2.INTER_LINEAR)
    else:
        image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    
    return image


# Test for one image 
if __name__ == "__main__":
    image_path = "entirety/obv/3984_a.jpg"
    image = cv2.imread(image_path)  #Load image 

    cropped_image = apply_circle_crop(image, img_size=224, percentage=0.95)
    grayscale_image = apply_grayscale(cropped_image, img_size=224, keep_ratio=True)

    # Save or display the processed images
    cv2.imshow("Cropped Image", cropped_image)
    cv2.imshow("Grayscale Image", grayscale_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    create_folder = True  # Set to False if you don't want to save the images
    if create_folder:
        import os
        
        input_folder = "entirety/obv"
        output_folder = "entirety/processed_images"
        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(input_folder):
            if filename.lower().endswith(".jpg"):
                image_path = os.path.join(input_folder, filename)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Fehler beim Laden von: {image_path}")
                    continue

                # Anwenden der Verarbeitung
                cropped_image = apply_circle_crop(image, img_size=224, percentage=0.95)
                grayscale_image = apply_grayscale(cropped_image, img_size=224, keep_ratio=True)

                # Bild speichern
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, grayscale_image)