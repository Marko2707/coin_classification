import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from Dataloader import train_loader, val_loader
from torch.utils.data import random_split
import torchvision.transforms.functional as TF

from Preprocessing import apply_circle_crop, apply_grayscale

#------------------------------------------------
# Set to True for grayscale images, False for color images!
use_grayscale = False  

#------------------------------------------------


class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # ohne FC
        self.embedding = nn.Linear(2048, 128)  # 128-dim Embedding

    def forward(self, x):
        x = self.feature_extractor(x)  # [B, 2048, 1, 1]
        x = x.view(x.size(0), -1)      # [B, 2048]
        x = self.embedding(x)          # [B, 128]
        x = nn.functional.normalize(x, p=2, dim=1)  # L2-Norm
        return x

"""Crops images in a circular shape and resizes them. NO GRAYSCALING"""
def preprocess_batch_crop_only(batch, img_size=224):
    batch_np = batch.permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, C]
    processed = []

    for img in batch_np:
        img = (img * 255).astype(np.uint8)
        img = apply_circle_crop(img, img_size=img_size)  # Output: [H, W, 3]
        # Kein expand_dims, sondern Channels an die erste Stelle bringen
        img = np.transpose(img, (2, 0, 1))  # [3, H, W]
        img = torch.tensor(img, dtype=torch.float32) / 255.0
        processed.append(img)

    return torch.stack(processed)  # [B, 3, H, W]


"""Crops images in a circular shape and resizes them with grayscale conversion."""
def preprocess_batch_grayscale(batch, img_size=224):
    batch_np = batch.permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, C]
    processed = []

    for img in batch_np:
        img = (img * 255).astype(np.uint8)
        img = apply_circle_crop(img, img_size=img_size)
        img = apply_grayscale(img, img_size=img_size)  # [H, W]

        # Expand dims auf (1, H, W), dann auf 3 Kanäle kopieren
        img = np.expand_dims(img, axis=0)     # [1, H, W]
        img = np.repeat(img, 3, axis=0)       # [3, H, W]

        img = torch.tensor(img, dtype=torch.float32) / 255.0
        processed.append(img)

    return torch.stack(processed)  # [B, 3, H, W]


def train_model():
    model = EmbeddingNet().to("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.TripletMarginLoss(margin=0.3)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        total_train_loss = 0
        for anchor, positive, negative in train_loader:

            if use_grayscale:   
                # Preprocess images for grayscale
                anchor = preprocess_batch_grayscale(anchor)
                positive = preprocess_batch_grayscale(positive)
                negative = preprocess_batch_grayscale(negative)
            else:
                # Preprocess images 
                anchor = preprocess_batch_crop_only(anchor)
                positive = preprocess_batch_crop_only(positive)
                negative = preprocess_batch_crop_only(negative)

            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss = criterion(emb_a, emb_p, emb_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)

                emb_a = model(anchor)
                emb_p = model(positive)
                emb_n = model(negative)

                val_loss = criterion(emb_a, emb_p, emb_n)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")
 
    model_name = "triplet_model.pth"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "modell", model_name)

    torch.save(model.state_dict(), model_path)



if __name__ == "__main__":
    train_model()
    