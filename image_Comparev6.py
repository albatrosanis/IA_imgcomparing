import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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
encoder.load_state_dict(torch.load('autoencoder_checkpoint.pth', map_location=device), strict=False)
encoder.eval()

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Encode a single image
def encode_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        latent_vector = encoder(image)
    return latent_vector.cpu().numpy()


# Precompute and save latent vectors for the dataset
def compute_and_save_latents(dataset_dir, latents_save_path):
    latents = []
    image_paths = []

    for file_name in tqdm(os.listdir(dataset_dir), desc="Encoding dataset"):
        image_path = os.path.join(dataset_dir, file_name)
        latent_vector = encode_image(image_path)
        latents.append(latent_vector)
        image_paths.append(image_path)

    latents = np.concatenate(latents, axis=0)  # Convert to a 2D NumPy array
    np.save(latents_save_path, latents)
    np.save(latents_save_path.replace(".npy", "_paths.npy"), image_paths)
    print(f"Latent representations and paths saved to {latents_save_path}")


def find_closest_image_with_latents(input_image_path, latents_save_path):
    # Allow pickle when loading latents
    latents = np.load(latents_save_path, allow_pickle=True)
    
    # Load the image paths with allow_pickle=True
    image_paths = np.load(latents_save_path.replace(".npy", "_paths.npy"), allow_pickle=True)

    input_latent = encode_image(input_image_path)

    distances = np.linalg.norm(latents - input_latent, axis=1)
    closest_index = np.argmin(distances)

    closest_image_path = image_paths[closest_index]
    min_distance = distances[closest_index]

    return closest_image_path, min_distance



# Display comparison between input and closest match
def display_comparison(input_image_path, closest_image_path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(Image.open(input_image_path))
    axs[0].set_title("Input Image")
    axs[0].axis("off")
    axs[1].imshow(Image.open(closest_image_path))
    axs[1].set_title("Closest Match")
    axs[1].axis("off")
    plt.show()


# Main execution
if __name__ == "__main__":
    dataset_dir = "C:/Users/Aures/Documents/GitHub/Train_Encode_Decode_IA/pokemon"
    latents_save_path = "dataset_latents.npy"
    input_image_path = "006.png"

    # Compute and save latents (run this once)
    if not os.path.exists(latents_save_path):
        compute_and_save_latents(dataset_dir, latents_save_path)

    # Find closest image
    closest_image, distance = find_closest_image_with_latents(input_image_path, latents_save_path)
    print(f"Closest image found: {closest_image} with distance: {distance:.4f}")

    # Display the results
    display_comparison(input_image_path, closest_image)
