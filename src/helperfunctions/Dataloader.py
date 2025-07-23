""" Dataloader for loading triplet images from a folder structure.
This module defines a custom dataset class for loading triplet images """
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class TripletFolderDataset(Dataset):
    """A dataset class for loading triplet images from a folder structure."""
    def __init__(self, triplet_dir, transform=None):
        self.triplet_paths = [] # List of dictionaries to hold triplet paths
        self.transform = transform
        for name in os.listdir(triplet_dir):
            path = os.path.join(triplet_dir, name)
            if os.path.isdir(path):
                files = os.listdir(path)

                # Finds all files that start with "Anchor", "Positive" and  "Negative"
                anchor_file = next((f for f in files if f.lower().startswith("anchor_")), None)
                positive_file = next((f for f in files if f.lower().startswith("positive_")), None)
                negative_file = next((f for f in files if f.lower().startswith("negative_")), None)

                if anchor_file and positive_file and negative_file:
                    self.triplet_paths.append({
                        'anchor': os.path.join(path, anchor_file),
                        'positive': os.path.join(path, positive_file),
                        'negative': os.path.join(path, negative_file),
                    })

    """Returns the number of triplets in the dataset."""
    def __len__(self):
        return len(self.triplet_paths)

    """Returns a triplet of images (anchor, positive, negative) at the given index."""
    def __getitem__(self, idx):
        paths = self.triplet_paths[idx]
        anchor = Image.open(paths["anchor"]).convert("RGB")
        positive = Image.open(paths["positive"]).convert("RGB")
        negative = Image.open(paths["negative"]).convert("RGB")
        # Applies transformations 
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        # Returns the triplet images
        return anchor, positive, negative

