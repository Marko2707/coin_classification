"""Module to preprocess data for training and evaluation."""
import cv2
import numpy as np

"""Apply a circular crop to an image.
This function reads an image from the specified path, applies a circular crop, and returns the cropped image. 
Original Source provided by our Tutors: https://github.com/Urjarm/DieStudyTool/blob/main/utils.py Original Author: Markus Fiedler
Modified for use under Windows and to return the cropped image directly.
"""
def apply_circle_crop(image, img_size=224, percentage=0.95, resize=True):
    # Wenn resize gewünscht, Bild auf img_size x img_size skalieren
    if resize:
        image = cv2.resize(image, (img_size, img_size))

    height, width = image.shape[:2]
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (width // 2, height // 2), int(max(height, width)/2 * percentage), 1, thickness=-1)
    
    # Maske auf alle Kanäle anwenden
    if len(image.shape) == 3 and image.shape[2] == 3:
        masked = cv2.bitwise_and(image, image, mask=circle_img)
    else:
        # Falls Bild schon grayscale ist
        masked = cv2.bitwise_and(image, image, mask=circle_img)
    
    return masked

"""Apply a grayscale transformation to an image.
This function reads an image from the specified path, converts it to grayscale, and resizes it to the specified size.
Original Source provided by our Tutors: https://github.com/Urjarm/DieStudyTool/blob/main/utils.py Original Author: Markus Fiedler

Modified for use under Windows and to return the cropped image directly.
"""
def apply_grayscale(image, img_size=224, keep_ratio=False):
    # In Graustufen konvertieren, falls noch nicht
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if keep_ratio:
        height, width = image.shape
        factor = float(img_size) / max(height, width)
        image = cv2.resize(image, (int(width*factor), int(height*factor)), interpolation=cv2.INTER_LINEAR)
    else:
        image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    
    return image


if __name__ == "__main__":
    image_path = "entirety/obv/3984_a.jpg"
    image = cv2.imread(image_path)  # Bild jetzt selbst laden

    cropped_image = apply_circle_crop(image, img_size=224, percentage=0.95)
    grayscale_image = apply_grayscale(cropped_image, img_size=224, keep_ratio=True)

    # Save or display the processed images
    cv2.imshow("Cropped Image", cropped_image)
    cv2.imshow("Grayscale Image", grayscale_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
