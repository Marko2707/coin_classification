import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import math
import hdbscan

# Modell laden: Vision Transformer (ViT)
model = models.vit_b_16(pretrained=True)
model.eval()

# Feature-Extraktion via cls_token
features = []
def hook_fn(module, input, output):
    cls_token = output[:, 0, :]  # Nur das erste Token extrahieren
    features.append(cls_token.detach())

model.encoder.layers[-1].ln_1.register_forward_hook(hook_fn)  # Entspricht norm1

# Bild-Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Bilder laden
image_dir = "E:/Uni/Master/DataChallanges/coin_classification/Dataset/obverse/A"
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
images = [transform(Image.open(p).convert("RGB")) for p in image_paths]

# Verarbeitung & Feature Extraction
embeddings = []
with torch.no_grad():
    for img in images:
        features.clear()
        _ = model(img.unsqueeze(0))
        embeddings.append(features[0].squeeze().numpy())

# Clustering mit HDBSCAN
X = np.array(embeddings)
clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=1, cluster_selection_method='eom', metric='euclidean', cluster_selection_epsilon=0.5)
labels = clusterer.fit_predict(X)

# Ergebnisse anzeigen
print("Cluster-Zuordnung der Münzbilder:\n")
for path, label in zip(image_paths, labels):
    print(f"Cluster {label}: {os.path.basename(path)}")

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_outliers = list(labels).count(-1)
print(f"\nGefundene Cluster: {n_clusters}")
print(f"Outlier (nicht zugeordnet): {n_outliers}")

# t-SNE
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

# Grad-CAM mit HiResCAM
target_layer = model.encoder.layers[-1].ln_1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
cam = HiResCAM(model=model, target_layers=[target_layer])

images_per_page = 10
num_pages = math.ceil(len(images) / images_per_page)

for page in range(num_pages):
    start_idx = page * images_per_page
    end_idx = min((page + 1) * images_per_page, len(images))
    subset = list(zip(images[start_idx:end_idx], image_paths[start_idx:end_idx]))

    plt.figure(figsize=(12, 3 * len(subset)))

    for i, (img_tensor, path) in enumerate(subset):
        input_tensor = img_tensor.unsqueeze(0).to(device)
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        
        img_pil = Image.open(path).convert("RGB").resize((224, 224))
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        plt.subplot(len(subset), 2, 2 * i + 1)
        plt.imshow(img_np)
        plt.axis("off")
        plt.title(f"Original: {os.path.basename(path)}")

        plt.subplot(len(subset), 2, 2 * i + 2)
        plt.imshow(visualization)
        plt.axis("off")
        plt.title("Grad-CAM")

    plt.tight_layout()
    plt.suptitle(f"Seite {page + 1}/{num_pages}", fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.show()
