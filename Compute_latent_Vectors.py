import torch
import os
import pickle
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# Define the Encoder (from your trained autoencoder)
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

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Function to encode and save dataset latent vectors
def preprocess_dataset(dataset_dir, encoder, device, output_file="latent_vectors.pkl"):
    latent_vectors = {}
    encoder.eval()
    with torch.no_grad():
        for file_name in tqdm(os.listdir(dataset_dir), desc="Encoding dataset"):
            image_path = os.path.join(dataset_dir, file_name)
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)
            latent_vector = encoder(image_tensor).cpu()
            latent_vectors[file_name] = latent_vector

    # Save latent vectors to a file
    with open(output_file, "wb") as f:
        pickle.dump(latent_vectors, f)
    print(f"Latent vectors saved to {output_file}")

# Main execution for preprocessing
if __name__ == "__main__":
    latent_dim = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained encoder
    encoder = Encoder(latent_dim).to(device)
    encoder.load_state_dict(torch.load("autoencoder_final.pth", map_location=device), strict=False)

    dataset_dir = "C:/Users/Aures/Documents/GitHub/Train_Encode_Decode_IA/pokemon"
    output_file = "latent_vectors.pkl"

    preprocess_dataset(dataset_dir, encoder, device, output_file)
