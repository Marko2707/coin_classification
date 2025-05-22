import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import math
import hdbscan
from collections import defaultdict
from GradCamVisualization import visualize_clusters, visualize_order  # Importiere die Funktion aus dem Plot-Skript

#Choose mode of visualization
mode = 1  # 1 == "ClusterVisualization" oder 2 == "OrderVisualization"

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
image_dir = "Dataset/obverse/Prot"
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

# Clustering mit HDBSCAN (automatische Clusteranzahl)
X = np.array(embeddings)
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_method='eom', metric='euclidean')
labels = clusterer.fit_predict(X)

# Print Cluster-Zuordnung
print("Cluster-Zuordnung der Münzbilder:\n")
for path, label in zip(image_paths, labels):
    print(f"Cluster {label}: {os.path.basename(path)}")

# Anzahl der Cluster und Outlier anzeigen
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_outliers = list(labels).count(-1)
print(f"\nGefundene Cluster: {n_clusters}")
print(f"Outlier (nicht zugeordnet): {n_outliers}")

# t-SNE für Visualisierung
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


# -----------------------------
target_layer = model.layer4[-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

cam = HiResCAM(model=model, target_layers=[target_layer])
if torch.cuda.is_available():
    model.to('cuda')


# Visualisierung der Cluster oder der Reihenfolge
if mode == 1:
    # Clusterweise Gruppieren
    clustered_images = defaultdict(list)
    for label, img_tensor, path in zip(labels, images, image_paths):
        clustered_images[label].append((img_tensor, path))

    visualize_clusters(clustered_images, cam, device)
elif mode == 2:
    visualize_order(images, image_paths, cam, device)

