import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from Dataloader import train_loader, val_loader
from torch.utils.data import random_split

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

model = EmbeddingNet().to("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.TripletMarginLoss(margin=0.2)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    total_train_loss = 0
    for anchor, positive, negative in train_loader:
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
