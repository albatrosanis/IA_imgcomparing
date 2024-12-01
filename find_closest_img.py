import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Define the Encoder (same as in your trained autoencoder)
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

# Load the encoder and trained weights
latent_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder(latent_dim).to(device)
encoder.load_state_dict(torch.load('autoencoder_final.pth', map_location=device), strict=False)
encoder.eval()

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Preprocess and encode an image
def encode_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Ensure 3 channels (RGB)
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        latent_vector = encoder(image)
    latent_vector = F.normalize(latent_vector, dim=1)  # Normalize latent vectors
    return latent_vector

# Find the closest image in a dataset
def find_closest_image(input_image_path, dataset_dir):
    input_latent = encode_image(input_image_path)

    closest_image_path = None
    min_distance = float('inf')

    for file_name in tqdm(os.listdir(dataset_dir), desc="Searching dataset"):
        image_path = os.path.join(dataset_dir, file_name)
        dataset_latent = encode_image(image_path)
        distance = torch.norm(input_latent - dataset_latent).item()  # Euclidean distance

        if distance < min_distance:
            min_distance = distance
            closest_image_path = image_path

    return closest_image_path, min_distance

# Find the top N closest images in a dataset
def find_top_closest_images(input_image_path, dataset_dir, top_n=10):
    input_latent = encode_image(input_image_path)
    closest_images = []

    for file_name in tqdm(os.listdir(dataset_dir), desc="Searching dataset"):
        image_path = os.path.join(dataset_dir, file_name)
        dataset_latent = encode_image(image_path)
        distance = torch.norm(input_latent - dataset_latent).item()  # Euclidean distance
        
        closest_images.append((image_path, distance))
        closest_images = sorted(closest_images, key=lambda x: x[1])[:top_n]

    return closest_images

# Display results for the closest image
def display_comparison(input_image_path, closest_image_path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(Image.open(input_image_path))
    axs[0].set_title("Input Image")
    axs[0].axis("off")
    axs[1].imshow(Image.open(closest_image_path))
    axs[1].set_title("Closest Match")
    axs[1].axis("off")
    plt.show()

# Display results for top N closest images
def display_top_comparisons(input_image_path, closest_images):
    fig, axs = plt.subplots(1, len(closest_images) + 1, figsize=(15, 5))
    axs[0].imshow(Image.open(input_image_path))
    axs[0].set_title("Input Image")
    axs[0].axis("off")
    
    for i, (image_path, distance) in enumerate(closest_images):
        axs[i + 1].imshow(Image.open(image_path))
        axs[i + 1].set_title(f"Match {i+1}\nDist: {distance:.4f}")
        axs[i + 1].axis("off")
    
    plt.show()

# Visualize the latent space using PCA
def visualize_latent_space(dataset_dir):
    latent_vectors = []
    image_names = []

    for file_name in os.listdir(dataset_dir):
        image_path = os.path.join(dataset_dir, file_name)
        latent_vector = encode_image(image_path).cpu().numpy().flatten()
        latent_vectors.append(latent_vector)
        image_names.append(file_name)

    latent_vectors = np.array(latent_vectors)
    pca = PCA(n_components=2)  # Reduce dimensions to 2 for plotting
    reduced_vectors = pca.fit_transform(latent_vectors)

    plt.figure(figsize=(10, 10))
    for i, image_name in enumerate(image_names):
        plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1], label=image_name, alpha=0.7)
    plt.title("Latent Space Visualization")
    plt.show()

# Example usage
if __name__ == "__main__":
    input_image_path = "006.png"
    dataset_dir = "C:/Users/Aures/Documents/GitHub/Train_Encode_Decode_IA/pokemon"

    print("Finding closest image...")
    closest_image_path, min_distance = find_closest_image(input_image_path, dataset_dir)
    print(f"Closest Image: {closest_image_path}, Distance: {min_distance:.4f}")
    display_comparison(input_image_path, closest_image_path)

    print("Finding top 10 closest images...")
    top_closest_images = find_top_closest_images(input_image_path, dataset_dir, top_n=10)
    for i, (image_path, distance) in enumerate(top_closest_images):
        print(f"Match {i+1}: {image_path} with distance: {distance:.4f}")
    display_top_comparisons(input_image_path, top_closest_images)

    print("Visualizing latent space...")
    visualize_latent_space(dataset_dir)
