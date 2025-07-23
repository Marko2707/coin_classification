import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super(EmbeddingNet, self).__init__()
        base_model = efficientnet_b0(pretrained=True)
        base_model.classifier = nn.Identity()  # removal of classification head
        self.base = base_model
        self.fc = nn.Linear(1280, embedding_dim)  # 1280 = Feature-Dim of efficientnet_b0

    def forward(self, x):
        x = self.base(x)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
