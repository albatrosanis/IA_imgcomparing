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

 #Ensure correct multiprocessing context
multiprocessing.set_start_method('spawn', force=True)

# Parameters
batch_size = 32
lr = 1e-4
epochs = 200
latent_dim = 128
noise_scale = 0.3
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
num_classes = 10  # Example: Adjust according to your dataset if classification is needed

# Dataset-specific transforms
def to_rgb(img):
    if img.mode != 'RGB':
        return img.convert('RGB')
    return img

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Lambda(to_rgb),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
custom_data_path = "C:/Users/Aures/Documents/GitHub/Train_Encode_Decode_IA/pokemon"
custom_dataset = CustomPokemonDataset(data_dir=custom_data_path, transform=transform)
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

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

# Initialize Model, Optimizer, and Loss
autoencoder = Autoencoder(latent_dim).to(device)
optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
criterion = nn.MSELoss()

# Checkpoint file path
checkpoint_path = 'autoencoder_checkpoint.pth'

# Load model from checkpoint
def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss_history = checkpoint['loss_history']
        print(f"Resuming training from epoch {epoch + 1}")
        return model, optimizer, epoch, loss_history
    else:
        print("No checkpoint found. Starting training from scratch.")
        return model, optimizer, 0, []

# Training Function
def train_autoencoder(model, dataloader, optimizer, criterion, total_epochs, start_epoch=0):
    loss_history = []
    model.train()
    
    # Train for the remaining epochs
    for epoch in range(start_epoch, total_epochs):
        running_loss = 0.0
        for data in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{total_epochs}'):
            inputs = data.to(device)

            # Add noise to inputs
            noisy_inputs = inputs + noise_scale * torch.randn_like(inputs)
            noisy_inputs = torch.clamp(noisy_inputs, -1., 1.)

            optimizer.zero_grad()
            outputs = model(noisy_inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{total_epochs}, Loss: {avg_loss:.4f}")

        # Save model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': loss_history
        }, checkpoint_path)

    return loss_history
# Function to preprocess the input image
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

    # Transform for preprocessing query image
    query_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load the trained model
    autoencoder = Autoencoder(latent_dim).to(device)
    autoencoder.load_state_dict(torch.load('autoencoder_final.pth'))

    # Find the most similar image
    most_similar, distance = find_most_similar_image(
        autoencoder, dataloader, query_image_path, query_transform, device
    )

    print(f"Minimum distance: {distance}")

    # Display the most similar image
    if most_similar is not None:
        most_similar = np.transpose(most_similar, (1, 2, 0))  # CHW to HWC
        most_similar = (most_similar * 0.5 + 0.5)  # Denormalize
        plt.imshow(most_similar)
        plt.title("Most Similar Image")
        plt.axis("off")
        plt.show()
    else:
        print("No similar image found.")
