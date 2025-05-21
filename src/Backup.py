import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import math

# Modell laden
model = models.resnet50(pretrained=True)
model.eval()

# Hook für Feature-Extraktion (layer4[-1])
features = []
def hook_fn(module, input, output):
    features.append(output.detach())

model.layer4[-1].register_forward_hook(hook_fn)

# Bild-Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Bilder laden
image_dir = "E:/Uni/Master/DataChallanges/coin_classification/Dataset/obverse/Prot"
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
images = [transform(Image.open(p).convert("RGB")) for p in image_paths]

# Verarbeitung & Feature Extraction
embeddings = []
with torch.no_grad():
    for img in images:
        features.clear()
        _ = model(img.unsqueeze(0))
        gap = torch.mean(features[0], dim=[2, 3])  # Global Average Pooling
        embeddings.append(gap.squeeze().numpy())

# Clustering mit KMeans
X = np.array(embeddings)
kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
labels = kmeans.labels_

# Print Cluster-Zuordnung
print("Cluster-Zuordnung der Münzbilder:\n")
for path, label in zip(image_paths, labels):
    print(f"Cluster {label}: {os.path.basename(path)}")

# t-SNE für Visualisierung
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="viridis")
plt.title("Clustering der Münzen mit ResNet50 + t-SNE")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()

# -----------------------------
# 🔍 Grad-CAM Visualisierung
# -----------------------------
target_layer = model.layer4[-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

cam = GradCAM(model=model, target_layers=[target_layer])
if torch.cuda.is_available():
    model.to('cuda')

# Anzeigen
images_per_page = 10
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