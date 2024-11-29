from tqdm import tqdm
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
import multiprocessing
from Custom_datasrt import CustomPokemonDataset  # Correct class name
import numpy as np


# Ensure correct multiprocessing context
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# Parameters
latent_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchvision import transforms

# Define your transforms
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),  # Convert RGBA to RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize for RGB
])
# Define Autoencoder Components
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (512, 8, 8)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)

class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# Function to preprocess the query image
def preprocess_image(image_path, transform):
    img = Image.open(image_path)
    img = transform(img)
    return img.unsqueeze(0)  # Add batch dimension

# Function to find the most similar image in the dataset
def find_most_similar_image(model, dataloader, query_img_path, transform, device):
    # Load and preprocess query image
    query_img = preprocess_image(query_img_path, transform).to(device)

    # Get the embedding of the query image
    model.eval()
    with torch.no_grad():
        query_embedding = model.encoder(query_img)

    min_distance = float('inf')
    most_similar_img = None

    # Iterate through the dataset to find the most similar image
    for data in tqdm(dataloader, desc="Finding Similar Image"):
        inputs = data.to(device)
        with torch.no_grad():
            dataset_embeddings = model.encoder(inputs)

        # Compute pairwise distances between the query and dataset embeddings
        distances = F.pairwise_distance(query_embedding, dataset_embeddings)

        # Find the closest match
        min_dist_idx = torch.argmin(distances).item()
        if distances[min_dist_idx] < min_distance:
            min_distance = distances[min_dist_idx]
            most_similar_img = inputs[min_dist_idx].cpu().numpy()

    return most_similar_img, min_distance

# Main function
if __name__ == "__main__":
    # Path to the query image
    query_image_path = "pikachu.jpg"
    from PIL import Image
    image = Image.open('pikachu.jpg')
    print(image.mode)  # Check the mode of the image (should print 'RGBA', 'RGB', etc.)


    # Path to the dataset
    custom_data_path = "C:/Users/Aures/Documents/GitHub/Train_Encode_Decode_IA/pokemon"
    custom_dataset = CustomPokemonDataset(data_dir=custom_data_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(custom_dataset, batch_size=32, shuffle=False)

    # Load the trained autoencoder model
    autoencoder = Autoencoder(latent_dim).to(device)
    autoencoder.load_state_dict(torch.load('autoencoder_final.pth'))

    # Find the most similar image
    most_similar, distance = find_most_similar_image(
        autoencoder, dataloader, query_image_path, transform, device
    )

    print(f"Minimum distance: {distance}")

    # Display the most similar image
    if most_similar is not None:
        most_similar = np.transpose(most_similar, (1, 2, 0))  # Convert from CHW to HWC format
        most_similar = (most_similar * 0.5 + 0.5)  # Denormalize the image
        plt.imshow(most_similar)
        plt.title("Most Similar Image")
        plt.axis("off")
        plt.show()
    else:
        print("No similar image found.")
