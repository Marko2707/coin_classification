import os
import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import hdbscan
from torchvision import transforms

from training_triplets import EmbeddingNet
from GradCamVisualization import visualize_clusters, visualize_order
from pytorch_grad_cam import GradCAM

from training_triplets import preprocess_batch_grayscale, preprocess_batch_crop_only  # deine beiden Funktionen

# ---------- SETTINGS ---------- #
use_grayscale_and_crop = True  # Setze False für nur Crop


# ---------- SETTINGS ---------- #
mode = 1  # 1 = ClusterVisualization, 2 = OrderVisualization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- MODEL LOADING ---------- #
model = EmbeddingNet().to(device)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "modell", "triplet_model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
print(f"Modell geladen: {model.__class__.__name__}")
model.eval()


# ---------- GRAD-CAM SETUP ---------- #
target_layer = model.feature_extractor[7][2].conv3
cam = GradCAM(model=model, target_layers=[target_layer])

# ---------- IMAGE LOADING ---------- #
# image_dir = os.path.join(current_dir, "..", "dataset", "obverse", "Prot")
image_dir = os.path.join(current_dir, "..", "entirety", "processed_images")
#image_dir = os.path.join(current_dir, "..", "entirety", "obv")
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]

image_tensors = []
for p in image_paths:
    img = Image.open(p).convert("RGB").resize((224, 224))
    img_tensor = transforms.ToTensor()(img)  # float tensor [3, 224, 224]
    image_tensors.append(img_tensor)

processed_batch = torch.stack(image_tensors)  # [B, 3, 224, 224]

# ---------- PREPROCESSING ---------- #
"""
if use_grayscale_and_crop:
    processed_batch = preprocess_batch_grayscale(batch, img_size=224)          # mit Grayscale + Crop
else:
    processed_batch = preprocess_batch_crop_only(batch, img_size=224)  # nur Crop

processed_batch = processed_batch.to(device)
"""
# ---------- FEATURE EXTRACTION ---------- #

embeddings = []
with torch.no_grad():
    for img_tensor in processed_batch:
        emb = model(img_tensor.unsqueeze(0).to(device))
        embeddings.append(emb.squeeze().cpu().numpy())

X = np.array(embeddings)

# ---------- CLUSTERING ---------- #
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2, cluster_selection_method='eom', metric='euclidean') #euclidean, cosine, manhattan, eom, leaf
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

# ---------- VISUALISIERUNG ---------- #
if mode == 1:
    clustered_images = defaultdict(list)
    for label, img_tensor, path in zip(labels, processed_batch, image_paths):
        clustered_images[label].append((img_tensor, path))
    visualize_clusters(clustered_images, cam, device)
elif mode == 2:
    visualize_order(processed_batch, image_paths, cam, device)
