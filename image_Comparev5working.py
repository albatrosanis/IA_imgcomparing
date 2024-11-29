import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch.nn as nn

# Ensure Encoder class is defined here or imported from another module
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

# Parameters
latent_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the encoder and trained weights
encoder = Encoder(latent_dim).to(device)
encoder.load_state_dict(torch.load('autoencoder_final.pth', map_location=device), strict=False)
encoder.eval()

# Transform pipeline
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Precompute latent vectors for the dataset
def preprocess_dataset(dataset_dir, encoder, transform, device):
    latents = {}
    encoder.eval()
    for file_name in tqdm(os.listdir(dataset_dir), desc="Encoding dataset"):
        image_path = os.path.join(dataset_dir, file_name)
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                latent = encoder(image_tensor).cpu().numpy()
            latents[file_name] = latent
        except Exception as e:
            print(f"Skipping {file_name}: {e}")
    return latents

# Save precomputed latents to a file
def save_latents(latents, save_path):
    np.save(save_path, latents)

# Load precomputed latents from a file
def load_latents(save_path):
    return np.load(save_path, allow_pickle=True).item()
# Function to encode a single input image
def encode_image(image_path, encoder, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        latent_vector = encoder(image)
    return latent_vector

# Match input image to dataset using precomputed latents
def find_closest_image_fast(input_image_path, dataset_latents, encoder, transform, device):
    input_latent = encode_image(input_image_path, encoder, transform, device)
    min_distance = float('inf')
    closest_image_name = None

    for file_name, dataset_latent in dataset_latents.items():
        dataset_latent_tensor = torch.tensor(dataset_latent, device=device)
        distance = torch.norm(input_latent - dataset_latent_tensor).item()

        if distance < min_distance:
            min_distance = distance
            closest_image_name = file_name

    return closest_image_name, min_distance
import matplotlib.pyplot as plt

# Display the input image and closest match
def display_comparison(input_image_path, closest_image_path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(Image.open(input_image_path))
    axs[0].set_title("Input Image")
    axs[0].axis("off")
    axs[1].imshow(Image.open(closest_image_path))
    axs[1].set_title("Closest Match")
    axs[1].axis("off")
    plt.show()

# Ensure Autoencoder, Encoder, and Decoder classes are defined here or imported from another module
# Include the training code for your autoencoder if needed
if __name__ == "__main__":
    # Paths
    input_image_path = "006.png"
    dataset_dir = "C:/Users/Aures/Documents/GitHub/Train_Encode_Decode_IA/pokemon"
    latents_save_path = "dataset_latents.npy"

    # Step 1: Precompute Latents
    if not os.path.exists(latents_save_path):
        print("Precomputing dataset latents...")
        dataset_latents = preprocess_dataset(dataset_dir, encoder, transform, device)
        save_latents(dataset_latents, latents_save_path)
        print(f"Latents saved to {latents_save_path}")
    else:
        print(f"Loading precomputed latents from {latents_save_path}...")
        dataset_latents = load_latents(latents_save_path)

    # Step 2: Find Closest Image
    closest_image_name, distance = find_closest_image_fast(
        input_image_path, dataset_latents, encoder, transform, device
    )

    closest_image_path = os.path.join(dataset_dir, closest_image_name)
    print(f"Closest image found: {closest_image_path} with distance: {distance:.4f}")

    # Step 3: Display Results
    display_comparison(input_image_path, closest_image_path)
