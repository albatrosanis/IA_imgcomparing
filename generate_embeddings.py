import torch
import numpy as np
from torch.utils.data import DataLoader
from Custom_datasrt import CustomPokemonDataset
from train_data_v6 import Autoencoder  # Replace this with the actual import path of your Autoencoder model

# Parameters
latent_dim = 128
data_path = "C:/Users/Aures/Documents/GitHub/Train_Encode_Decode_IA/pokemon"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms (consistent with training)
from torchvision import transforms
def to_rgb(img):
    if img.mode == 'RGBA':
        img = img.convert('RGB')  # Convert RGBA to RGB
    elif img.mode != 'RGB':
        img = img.convert('RGB')  # Convert other modes (e.g., grayscale) to RGB
    return img

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Lambda(to_rgb),  # Ensure all images are RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])



# Load dataset and model
dataset = CustomPokemonDataset(data_dir=data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

autoencoder = Autoencoder(latent_dim).to(device)
autoencoder.load_state_dict(torch.load("autoencoder_final.pth", map_location=device))
autoencoder.eval()

# Generate embeddings
def generate_and_save_embeddings(model, dataloader, device, save_path="embeddings.npy"):
    embeddings = []
    with torch.no_grad():
        for data in dataloader:
            inputs = data.to(device)
            latent_vectors = model.encoder(inputs)
            embeddings.append(latent_vectors.cpu().numpy())
    embeddings = np.vstack(embeddings)
    np.save(save_path, embeddings)  # Save the embeddings to a file
    print(f"Embeddings saved to {save_path}")

generate_and_save_embeddings(autoencoder, dataloader, device)
