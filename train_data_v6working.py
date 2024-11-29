import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from Custom_datasrt import CustomPokemonDataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Parameters
batch_size = 32
lr = 1e-4
epochs = 50
latent_dim = 128
noise_scale = 0.3
checkpoint_path = "autoencoder_checkpoint.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
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

# Dataset and DataLoader
data_path = "C:/Users/Aures/Documents/GitHub/Train_Encode_Decode_IA/pokemon"
dataset = CustomPokemonDataset(data_dir=data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (512, 8, 8)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)

# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# Initialize model, optimizer, loss, and scaler
autoencoder = Autoencoder(latent_dim).to(device)
optimizer = optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=1e-5)
criterion = nn.MSELoss()
scaler = GradScaler()

# Load checkpoint if available
def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss_history = checkpoint['loss_history']
        print(f"Resuming from epoch {start_epoch + 1}")
        return model, optimizer, start_epoch, loss_history
    else:
        print("No checkpoint found. Starting training from scratch.")
        return model, optimizer, 0, []

# Save checkpoint
def save_checkpoint(model, optimizer, epoch, loss_history, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history
    }, path)
    print(f"Checkpoint saved at epoch {epoch + 1}")

# Training loop
def train_autoencoder(model, dataloader, optimizer, criterion, scaler, total_epochs, start_epoch=0):
    loss_history = []
    for epoch in range(start_epoch, total_epochs):
        model.train()
        running_loss = 0.0
        for data in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{total_epochs}"):
            inputs = data.to(device)
            noisy_inputs = inputs + noise_scale * torch.randn_like(inputs)
            noisy_inputs = torch.clamp(noisy_inputs, -1.0, 1.0)

            optimizer.zero_grad()
            with autocast():
                outputs = model(noisy_inputs)
                loss = criterion(outputs, inputs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{total_epochs}, Loss: {avg_loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, loss_history, checkpoint_path)

    return loss_history

# Main execution
if __name__ == "__main__":
    autoencoder, optimizer, start_epoch, loss_history = load_checkpoint(autoencoder, optimizer, checkpoint_path)

    loss_history = train_autoencoder(autoencoder, dataloader, optimizer, criterion, scaler, epochs, start_epoch)

    # Plot training loss
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_loss.png")
    plt.show()

    # Save final model
    torch.save(autoencoder.state_dict(), "autoencoder_final.pth")
    print("Final model saved as 'autoencoder_final.pth'")
