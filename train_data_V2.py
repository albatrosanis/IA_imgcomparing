import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
import multiprocessing
from Custom_datasrt import CustomPokemonDataset  # Ensure this is the correct path/class name
import argparse

# Ensure correct multiprocessing context
multiprocessing.set_start_method('spawn', force=True)

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--reset", action="store_true", help="Start training from scratch")
args = parser.parse_args()

# Parameters
batch_size = 32
lr = 1e-4
epochs = 220
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

# Load Dataset
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

# Main Execution
if __name__ == "__main__":
    if args.reset:
        print("Resetting training. Starting from scratch.")
        autoencoder, optimizer, start_epoch, loss_history = autoencoder, optimizer, 0, []
    else:
        autoencoder, optimizer, start_epoch, loss_history = load_checkpoint(autoencoder, optimizer, checkpoint_path)

    # Train Autoencoder
    loss_history = train_autoencoder(autoencoder, dataloader, optimizer, criterion, epochs, start_epoch)

    # Plot Loss
    plt.plot(loss_history, label="Reconstruction Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.savefig('loss_plot.png')  # Save the plot for later reference

    # Save Final Model
    torch.save(autoencoder.state_dict(), 'autoencoder_final.pth')
