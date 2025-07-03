import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from collections import defaultdict
from torchvision import transforms
from torchvision import models
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import hdbscan
from pytorch_grad_cam import HiResCAM, GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from training_triplets import EmbeddingNet
from GradCamVisualization import visualize_clusters, visualize_order

# ---------- SETTINGS ---------- #
mode = 1  # 1 = ClusterVisualization, 2 = OrderVisualization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- MODEL LOADING ---------- #
model = EmbeddingNet().to(device)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "modell", "triplet_model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ---------- GRAD-CAM SETUP ---------- #
# target_layer = model.feature_extractor[-1]
target_layer = model.feature_extractor[7][2].conv3
#cam = HiResCAM(model=model, target_layers=[target_layer])
cam = GradCAM(model=model, target_layers=[target_layer])
print(model.feature_extractor)

# ---------- IMAGE LOADING ---------- #
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

image_dir = os.path.join(current_dir, "..", "entirety", "obv")
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
images = [transform(Image.open(p).convert("RGB")) for p in image_paths]

# ---------- FEATURE EXTRACTION ---------- #
embeddings = []
with torch.no_grad():
    for img in images:
        emb = model(img.unsqueeze(0).to(device))
        embeddings.append(emb.squeeze().cpu().numpy())

X = np.array(embeddings)

# ---------- CLUSTERING ---------- #
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_method='eom', metric='euclidean')
labels = clusterer.fit_predict(X)

print("Cluster-Zuordnung der Münzbilder:\n")
for path, label in zip(image_paths, labels):
    print(f"Cluster {label}: {os.path.basename(path)}")

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_outliers = list(labels).count(-1)
print(f"\nGefundene Cluster: {n_clusters}")
print(f"Outlier (nicht zugeordnet): {n_outliers}")

# ---------- t-SNE VISUALIZATION ---------- #
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X)

plt.figure(figsize=(10, 7))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=40)
plt.title("HDBSCAN-Clustering der Münzen (t-SNE Visualisierung)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()

# Visualisierung der Cluster oder der Reihenfolge
if mode == 1:
    # Clusterweise Gruppieren
    clustered_images = defaultdict(list)
    for label, img_tensor, path in zip(labels, images, image_paths):
        clustered_images[label].append((img_tensor, path))

    visualize_clusters(clustered_images, cam, device)
elif mode == 2:
    visualize_order(images, image_paths, cam, device)
