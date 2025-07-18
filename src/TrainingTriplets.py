import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from helperfunctions.Dataloader import TripletFolderDataset
import matplotlib.pyplot as plt
from model import EmbeddingNet
from torch.utils.data import random_split

# --- Configuration ------------------------------------|

# Path for training and validation data
# Ovverse Images:
#train_data_path = "Trainingsset/obverse/cropped_grayscale" 

#Reverse Images:
#train_data_path = "Trainingsset/reverse/cropped_grayscale"
train_data_path = "Trainingsset/reverse/cropped"  # For cropped images

# Test/Validation Split
val_split = 0.8

# Resoulution for processing
img_size = 224  # Image Resulution for processing

# Batch Size:
batch_size = 16  # Batch Size for training and validation

# epochs 
epochs = 100  # Number of epochs for training
#epochs = 50

# We used both the pretrained and untrained resnet50 model --> To change that go to the model.py file  
#model_name = "resnet_50_pretrained_" + "rev_" + "crop-grayscale_" + str(epochs) + "-epochs.pth" # Name of the model to save
model_name = "resnet_50_pretrained_" + "rev_" + "crop_" + str(epochs) + "-epochs.pth" # Name of the model to save
#model_name = "resnet_50_pretrained_" + "rev_" + "normal_" + str(epochs) + "-epochs.pth" # Name of the model to save

#model_name = "resnet_50_untrained_" + "rev_" + "crop-grayscale_" + str(epochs) + "-epochs.pth" # Name of the model to save
#model_name = "resnet_50_untrained_" + "rev_" + "crop_" + str(epochs) + "-epochs.pth" # Name of the model to save
#model_name = "resnet_50_untrained_" + "rev_" + "normal_" + str(epochs) + "-epochs.pth" # Name of the model to save
#------------------------------------------------------|


# 1 . Transformation 
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),  
    #transforms.Resize((348, 348)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 2. Load Dataset
dataset = TripletFolderDataset(train_data_path, transform=transform)

# 3. Aufteilen in Training und Validierung
train_size = int(val_split * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 4. DataLoader 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



def show_image(tensor_img, title="Grayscale Image"):
    img = tensor_img.cpu().numpy()[0]  
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def train_model():
    """Train the triplet network model."""
    model = EmbeddingNet(embedding_dim=128).to("cuda" if torch.cuda.is_available() else "cpu")
    #model = EmbeddingNet().to("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    

    criterion = nn.TripletMarginLoss(margin=0.3)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # lr = learning rate 

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for anchor, positive, negative in train_loader:
           
            #show_image(anchor[0], title=f"Epoch {epoch+1} - Anchor Image") # To visualize the anchor image
            #break 

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
 

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "modell", model_name)

    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    train_model()
    