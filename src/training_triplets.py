import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from Dataloader import dataloader 

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

for epoch in range(10):  # z. B. 10 Epochen
    model.train()
    total_loss = 0
    for anchor, positive, negative in dataloader:
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

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}: Avg. Triplet Loss = {avg_loss:.4f}")
