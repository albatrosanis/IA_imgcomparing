import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
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
 
# Display results
def display_comparison(input_image_path, closest_image_path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(Image.open(input_image_path))
    axs[0].set_title("Input Image")
    axs[0].axis("off")
    axs[1].imshow(Image.open(closest_image_path))
    axs[1].set_title("Closest Match")
    axs[1].axis("off")
    plt.show()

# Example usage
if __name__ == "__main__":
    input_image_path = "006.png"
    dataset_dir = "C:/Users/Aures/Documents/GitHub/Train_Encode_Decode_IA/pokemon"

    closest_image, distance = find_closest_image(input_image_path, dataset_dir)
    print(f"Closest image found: {closest_image} with distance: {distance:.4f}")
    display_comparison(input_image_path, closest_image)

