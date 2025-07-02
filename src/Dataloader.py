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

                # Find files that start with "Anchor_", "Positive_", "Negative_"
                anchor_file = next((f for f in files if f.lower().startswith("anchor_")), None)
                positive_file = next((f for f in files if f.lower().startswith("positive_")), None)
                negative_file = next((f for f in files if f.lower().startswith("negative_")), None)

                if anchor_file and positive_file and negative_file:
                    self.triplet_paths.append({
                        'anchor': os.path.join(path, anchor_file),
                        'positive': os.path.join(path, positive_file),
                        'negative': os.path.join(path, negative_file),
                    })

    def __len__(self):
        return len(self.triplet_paths)

    def __getitem__(self, idx):
        paths = self.triplet_paths[idx]

        anchor = Image.open(paths["anchor"]).convert("RGB")
        positive = Image.open(paths["positive"]).convert("RGB")
        negative = Image.open(paths["negative"]).convert("RGB")

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative


#Beispiel-Transformation

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#Lade den Datensatz
dataset = TripletFolderDataset("Trainingsset", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print(f"Loaded {len(dataset)} triplets.")
print("Example paths from the first triplet:")

#Pfade des ersten Triplets anzeigen
if dataset.triplet_paths:
    paths = dataset.triplet_paths[0]
    print(f"  Anchor path:   {paths['anchor']}")
    print(f"  Positive path: {paths['positive']}")
    print(f"  Negative path: {paths['negative']}")
else:
    print("⚠️  Keine Triplet-Dateien gefunden!")

# Teste den DataLoader mit Ausgabe
print("\n🔍 Loading 1 batch from the DataLoader...\n")
for batch_idx, (anchors, positives, negatives) in enumerate(dataloader):
    print(f"🟢 Batch {batch_idx + 1}")
    print(f"  Anchor shape:   {anchors.shape} | dtype: {anchors.dtype}")
    print(f"  Positive shape: {positives.shape}")
    print(f"  Negative shape: {negatives.shape}")
    print(f"  Anchor min/max: {anchors.min().item():.2f} / {anchors.max().item():.2f}")
    break  # Nur ersten Batch anzeigen