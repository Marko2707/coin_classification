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
from collections import defaultdict
import hdbscan  # Import HDBSCAN
from GradCamVisualization import visualize_clusters, visualize_order  # Importiere die Funktion aus dem Plot-Skriptq 


# Modell laden
# model = models.efficientnet_b0(pretrained=True)  # EfficientNet-B0
model = models.resnet50(pretrained=True)  # ResNet-50
model.eval()

# Target-Layer für Grad-CAM
# target_layer = model.features[-1]  # EfficientNet-B0
target_layer = model.layer4[-1]  # ResNet-50

# Feature-Hook
features = []
def hook_fn(module, input, output):
    features.append(output.detach())

# model.features[-1].register_forward_hook(hook_fn)  # EfficientNet-B0
model.layer4[-1].register_forward_hook(hook_fn)  # ResNet-50

# Bild-Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Bilder laden
image_dir = "dataset/obverse/Prot"
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
images = [transform(Image.open(p).convert("RGB")) for p in image_paths]

# Feature-Extraktion
embeddings = []
with torch.no_grad():
    for img in images:
        features.clear()
        _ = model(img.unsqueeze(0))
        # gap = torch.mean(features[0], dim=[2, 3])  # Global Average Pooling (EfficientNet-B0)
        gap = torch.mean(features[0], dim=[2, 3])  # Global Average Pooling (ResNet-50, bleibt gleich)
        embeddings.append(gap.squeeze().numpy())

# KMeans-Clustering
X = np.array(embeddings)
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_method= "eom", metric='euclidean')  # Parameter anpassbar
labels = clusterer.fit_predict(X)

# Cluster-Zuordnung ausgeben
print("Cluster-Zuordnung der Münzbilder:\n")
for path, label in zip(image_paths, labels):
    print(f"Cluster {label}: {os.path.basename(path)}")

# t-SNE Visualisierung
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="viridis")
plt.title("Clustering der Münzen mit ResNet-50 + t-SNE")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()

# -----------------------------
# 🔍 Grad-CAM Visualisierung pro Cluster
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

cam = GradCAM(model=model, target_layers=[target_layer])

# Clusterweise Gruppieren
clustered_images = defaultdict(list)
for label, img_tensor, path in zip(labels, images, image_paths):
    clustered_images[label].append((img_tensor, path))

visualize_clusters(clustered_images, cam, device)
