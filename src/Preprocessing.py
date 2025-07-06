"""Module to preprocess data for training and evaluation."""
import cv2
import numpy as np

"""Apply a circular crop to an image.
This function reads an image from the specified path, applies a circular crop, and returns the cropped image. 
Original Source provided by our Tutors: https://github.com/Urjarm/DieStudyTool/blob/main/utils.py Original Author: Markus Fiedler
Modified for use under Windows and to return the cropped image directly.
"""
def apply_circle_crop(src, img_size=224, percentage=0.95, resize=True):
    image = cv2.imread(src)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {src}")

    if resize:
        image = cv2.resize(image, (img_size, img_size)) 

    height, width, _ = image.shape
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (width // 2, height // 2),
               int(max(height, width) / 2 * percentage), 1, thickness=-1)
    image = cv2.bitwise_and(image, image, mask=circle_img)

    return image

"""Apply a grayscale transformation to an image.
This function reads an image from the specified path, converts it to grayscale, and resizes it to the specified size.
Original Source provided by our Tutors: https://github.com/Urjarm/DieStudyTool/blob/main/utils.py Original Author: Markus Fiedler

Modified for use under Windows and to return the cropped image directly.
"""
def apply_grayscale(src, img_size=224, keep_ratio=False):
    if isinstance(src, str):
        image = cv2.imread(src)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {src}")
    else:
        image = src  # Assume it's a numpy array

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if keep_ratio:
        height, width = image.shape
        factor = float(img_size) / max(height, width)
        image = cv2.resize(image, (int(width * factor), int(height * factor)))
    else:
        image = cv2.resize(image, (img_size, img_size))

    return image


if __name__ == "__main__":
    # Example usage
    image_path = "entirety/obv/3984_a.jpg"
    cropped_image = apply_circle_crop(image_path, img_size=224, percentage=0.95)
    grayscale_image = apply_grayscale(cropped_image, img_size=224, keep_ratio=True)

    # Save or display the processed images
    cv2.imshow("Cropped Image", cropped_image)
    cv2.imshow("Grayscale Image", grayscale_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
