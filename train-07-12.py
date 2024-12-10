import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn as nn

# Ensure your model structure matches the trained one
from train_data_v7 import Encoder, Decoder  # Importing encoder/decoder from your module

# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

# Optimized Dataset class
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image), img_path

# Optimized function to compute latent vectors
def compute_latent_vectors(dataset_path, model, transform, device, batch_size=32):
    image_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(dataset_path)
        for file in files if file.endswith(('.png', '.jpg', '.jpeg'))
    ]

    dataset = ImageDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    latent_vectors = []
    all_image_paths = []

    for images, paths in tqdm(dataloader, desc="Computing latent vectors"):
        images = images.to(device)
        with torch.no_grad():
            latent = model.encoder(images).cpu().numpy()
        latent_vectors.append(latent)
        all_image_paths.extend(paths)
    
    latent_vectors = np.vstack(latent_vectors)
    return latent_vectors, all_image_paths

# Function to find the closest matching image
def find_closest_image(query_image_path, latent_vectors, image_paths, model, transform, device):
    # Load and preprocess query image
    query_image = Image.open(query_image_path).convert('RGB')
    query_image = transform(query_image).unsqueeze(0).to(device)

    # Compute the query image's latent vector
    with torch.no_grad():
        query_latent = model.encoder(query_image).cpu().numpy().flatten()

    # Compute distances
    distances = np.linalg.norm(latent_vectors - query_latent, axis=1)

    # Find the closest match
    closest_index = np.argmin(distances)
    closest_image_path = image_paths[closest_index]
    return closest_image_path, distances[closest_index]

# Main script
if __name__ == "__main__":
    # Set paths and parameters
    dataset_path = "C:/Users/Aures/Documents/GitHub/Train_Encode_Decode_IA/pokemon"
    query_image_path = "006.png"
    checkpoint_path = "autoencoder_checkpoint.pth"
    latent_dim = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained autoencoder
    autoencoder = Autoencoder(latent_dim).to(device)
    autoencoder.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    autoencoder.eval()

    # Image transformations (must match training transformations)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Match input size during training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Step 1: Compute latent vectors for all dataset images
    latent_vectors, image_paths = compute_latent_vectors(dataset_path, autoencoder, transform, device)

    # Step 2: Save latent vectors and image paths for reuse
    np.savez("latent_vectors.npz", latent_vectors=latent_vectors, image_paths=image_paths)

    # Step 3: Find the closest match to the query image
    closest_image_path, distance = find_closest_image(query_image_path, latent_vectors, image_paths, autoencoder, transform, device)

    print(f"Closest match: {closest_image_path} (Distance: {distance:.4f})")

    # Optional: Display the query and closest match images
    query_image = Image.open(query_image_path)
    closest_image = Image.open(closest_image_path)

    query_image.show(title="Query Image")
    closest_image.show(title="Closest Match")
