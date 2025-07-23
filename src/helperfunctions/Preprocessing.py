""" Preprocessing functions for image data, including circular cropping and grayscale conversion."""
import cv2
import numpy as np

""" Apply a circular crop to an image. Normal Circle and not perfect crop. --> So that model does not learn the shape of the coin.
    Original Source provided by our Tutors: https://github.com/Urjarm/DieStudyTool/blob/main/utils.py Original Author: Markus Fiedler
    Modified for use under Windows and to return the cropped image directly.
"""
def apply_circle_crop(image, img_size=224, percentage=0.95, resize=True, neutral_color=(123, 117, 104)):
    """
    image: input image as a BGR np.array of type uint8  
    img_size: target size (square)  
    percentage: defines how large the circle should be relative to the image (1.0 = full radius)  
    resize: whether to scale the image to img_size before processing  
    neutral_color: color value (BGR) used to fill the area outside the circle  
    
    Returns the image with the area outside the circle filled with neutral_color. """
    if resize:
        image = cv2.resize(image, (img_size, img_size))

    height, width = image.shape[:2]

    # Creade a circular mask
    mask = np.zeros((height, width), dtype=np.uint8)
    radius = int(max(height, width) / 2 * percentage)
    center = (width // 2, height // 2)
    cv2.circle(mask, center, radius, 255, thickness=-1)

    # Apply the mask to the image
    masked_image = image.copy()

    # Set to neutral color outside of the circle 
    outside_indices = mask == 0
    for c in range(3):  #
        masked_image[:, :, c][outside_indices] = neutral_color[c]

    return masked_image


"""Apply a grayscale transformation to an image .
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
        image = cv2.resize(image, (int(width*factor), int(height*factor)), interpolation=cv2.INTER_LINEAR) # Resize while keeping aspect ratio
    else:
        image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    
    return image


# Test for one image if both functions work correctly
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
