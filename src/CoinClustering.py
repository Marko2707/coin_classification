import os
import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import hdbscan
from torchvision import transforms

from TrainingTriplets import EmbeddingNet
from helperfunctions.GradCamVisualization import visualize_clusters, visualize_order
from pytorch_grad_cam import GradCAM


# ---Configuration------------------------------------|
# GradCam Visualisation:
mode = 1  # 1 = ClusterVisualization, 2 = OrderVisualization

# Image Directory:
current_dir = os.path.dirname(os.path.abspath(__file__)) # Dont Change this line

# Choose the images you want to cluster:
#image_dir = os.path.join(current_dir, "..", "entirety", "obv_processed_images") # The obverse images (Preprocessed)
image_dir = os.path.join(current_dir, "..", "entirety", "rev_processed_images") # The reverse images (Preprocessed)

# If you want to use the original images, uncomment the following line (NOT RECOMMENDED --> Worse Results):
#image_dir = os.path.join(current_dir, "..", "entirety", "obv")
#image_dir = os.path.join(current_dir, "..", "entirety", "rev")

# Model Path, Name and Type:
model_name = "resnet_50_pretrained_rev_crop-grayscale_100-epochs.pth"  # Name of the model to load

# Clustering Configuration:
min_cluster_size = 2  # Minimum size of clusters 
min_samples = 2  # Minimum number of samples in a neighborhood for a point to be considered a core point
metric = 'manhattan'  # Distance metric for clustering (RECOMMENDED: 'manhattan', 'euclidean' gave worse results)
cluster_selection_method = 'leaf'  # Method to select clusters --> leaf is recommended. eom gave way worse results 
# ---------------------------------------------------|



# ---------- MODEL LOADING ---------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = EmbeddingNet().to(device)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "modell", model_name)
model.load_state_dict(torch.load(model_path, map_location=device))
print(f"Modell geladen: {model.__class__.__name__}")
model.eval()


# ---------- GRAD-CAM SETUP ---------- #
target_layer = model.feature_extractor[7][2].conv3 # Last convolutional layer of the modified ResNet50
cam = GradCAM(model=model, target_layers=[target_layer])


# ---------- IMAGE LOADING ---------- #
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
image_tensors = []
for p in image_paths:
    img = Image.open(p).convert("RGB").resize((224, 224))
    img_tensor = transforms.ToTensor()(img)  # float tensor [3, 224, 224]
    image_tensors.append(img_tensor)
processed_batch = torch.stack(image_tensors)  # [B, 3, 224, 224]


# ---------- EMBEDDING EXTRACTION ---------- #
embeddings = []
with torch.no_grad():
    for img_tensor in processed_batch:
        emb = model(img_tensor.unsqueeze(0).to(device))
        embeddings.append(emb.squeeze().cpu().numpy())
X = np.array(embeddings)

# ---------- CLUSTERING ---------- #
clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                            min_samples=min_samples, cluster_selection_method=cluster_selection_method, metric=metric)
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

# ---------- Cluster Visualization ---------- #
if mode == 1:
    clustered_images = defaultdict(list)
    for label, img_tensor, path in zip(labels, processed_batch, image_paths):
        clustered_images[label].append((img_tensor, path))
    visualize_clusters(clustered_images, cam, device)
elif mode == 2:
    visualize_order(processed_batch, image_paths, cam, device)
