import math
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
from collections import defaultdict

"""
clustered_images = defaultdict(list)
for label, img_tensor, path in zip(labels, images, image_paths):
    clustered_images[label].append((img_tensor, path))
"""

def visualize_clusters(clustered_images, cam, device, images_per_page=10,):
    """
    Visualizes clusters with Grad-CAM heatmaps.

    Parameters:
        clustered_images (dict): Dictionary where keys are cluster labels and values are lists of (image_tensor, path).
        cam (GradCAM): Grad-CAM object for generating heatmaps.
        device (torch.device): Device to run the model on (CPU or GPU).
        images_per_page (int): Number of images to display per page.
    """

    for cluster_label in sorted(clustered_images.keys()):
        subset = clustered_images[cluster_label]
        num_images = len(subset)
        num_pages = math.ceil(num_images / images_per_page)

        for page in range(num_pages):
            start_idx = page * images_per_page
            end_idx = min((page + 1) * images_per_page, num_images)
            current_subset = subset[start_idx:end_idx]

            plt.figure(figsize=(12, 3 * len(current_subset)))

            for i, (img_tensor, path) in enumerate(current_subset):
                input_tensor = img_tensor.unsqueeze(0).to(device)
                grayscale_cam = cam(input_tensor=input_tensor)[0]

                # Original image
                img_pil = Image.open(path).convert("RGB").resize((224, 224))
                img_np = np.array(img_pil).astype(np.float32) / 255.0
                visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

                plt.subplot(len(current_subset), 2, 2 * i + 1)
                plt.imshow(img_np)
                plt.axis("off")
                plt.title(f"Original: {os.path.basename(path)}")

                plt.subplot(len(current_subset), 2, 2 * i + 2)
                plt.imshow(visualization)
                plt.axis("off")
                plt.title("Grad-CAM")

            plt.tight_layout()
            plt.suptitle(f"Cluster {cluster_label} – Seite {page + 1}/{num_pages}", fontsize=16)
            plt.subplots_adjust(top=0.95)
            plt.show()

def visualize_order(images, image_paths, cam, device, images_per_page = 10):
    num_pages = math.ceil(len(images) / images_per_page)

    for page in range(num_pages):
        start_idx = page * images_per_page
        end_idx = min((page + 1) * images_per_page, len(images))
        subset = list(zip(images[start_idx:end_idx], image_paths[start_idx:end_idx]))

        plt.figure(figsize=(12, 3 * len(subset)))  # Breite anpassen nach Bedarf

        for i, (img_tensor, path) in enumerate(subset):
            input_tensor = img_tensor.unsqueeze(0).to(device)
            grayscale_cam = cam(input_tensor=input_tensor)[0]

            # Originalbild vorbereiten
            img_pil = Image.open(path).convert("RGB").resize((224, 224))
            img_np = np.array(img_pil).astype(np.float32) / 255.0
            visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

            # Originalbild
            plt.subplot(len(subset), 2, 2 * i + 1)
            plt.imshow(img_np)
            plt.axis("off")
            plt.title(f"Original: {os.path.basename(path)}")

            # Grad-CAM Overlay
            plt.subplot(len(subset), 2, 2 * i + 2)
            plt.imshow(visualization)
            plt.axis("off")
            plt.title("Grad-CAM")

        plt.tight_layout()
        plt.suptitle(f"Seite {page + 1}/{num_pages}", fontsize=16)
        plt.subplots_adjust(top=0.95)  # Platz für Titel
        plt.show()