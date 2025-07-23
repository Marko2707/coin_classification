""" EmbeddingNet class for generating image embeddings using a pre-trained or not pretrained ResNet50 model."""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        #base_model = models.resnet50(pretrained=False)
        base_model = models.resnet50(pretrained=True) 


        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # without the final FC layer since we want embeddings

        self.embedding = nn.Linear(2048, embedding_dim)

    def forward(self, x):
        # Pass the image through the ResNet feature extractor (without the final classification layer).
        # Output shape: [Batch size, 2048, 1, 1] – a compact feature map for each image.
        x = self.feature_extractor(x)

        # Flatten the feature map to remove the extra dimensions.
        # Resulting shape: [Batch size, 2048] – one long vector per image.
        x = x.view(x.size(0), -1)

        # Pass the 2048-dimensional vector through a linear layer to reduce it to 128 dimensions.
        # This is our embedding – a compact representation of the image.
        x = self.embedding(x)

        # Normalize the 128-dimensional vector to have length 1 (L2 norm).
        # This ensures that only the direction of the vector matters, not its size.
        x = nn.functional.normalize(x, p=2, dim=1)

        # Return the final 128-dimensional normalized embedding.
        return x