import torch
import torch.nn as nn
import torchvision.models as models

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
