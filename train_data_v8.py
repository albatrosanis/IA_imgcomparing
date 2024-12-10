import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import vgg16
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from Custom_datasrt import CustomPokemonDataset
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Parameters
batch_size = 64
lr = 1e-4
epochs = 20
latent_dim = 256
noise_scale = 0.3
checkpoint_path = "autoencoder_checkpoint.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
def to_rgb(img):
    if img.mode != 'RGB':
        return img.convert('RGB')
    return img

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Lambda(to_rgb),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomCrop((224, 224)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Dataset and DataLoader
data_path = "C:/Users/Aures/Documents/GitHub/Train_Encode_Decode_IA/pokemon"
assert os.path.exists(data_path), "Data directory does not exist."

dataset = CustomPokemonDataset(data_dir=data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                        num_workers=4, pin_memory=torch.cuda.is_available())

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
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Ensuring a consistent output size before flattening
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (1024, 4, 4)),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
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

# Perceptual Loss
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

    def forward(self, x, y):
        x_features = self.vgg(x)
        y_features = self.vgg(y)

        # Resize y_features to match x_features if necessary
        if x_features.shape != y_features.shape:
            y_features = nn.functional.interpolate(y_features, 
                                                   size=x_features.shape[2:], 
                                                   mode='bilinear', 
                                                   align_corners=False)
        return torch.mean((x_features - y_features) ** 2)
    
# Save checkpoint
def save_checkpoint(model, optimizer, epoch, loss_history, path, additional_info=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
        'additional_info': additional_info
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch + 1}")

# Load checkpoint
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

# Save reconstruction sample
def save_reconstruction_sample(model, dataloader, epoch, device, output_dir="reconstruction_samples"):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if idx > 0:  # Save only the first batch
                break
            inputs = data.to(device)
            outputs = model(inputs)
            inputs = (inputs * 0.5 + 0.5).cpu().numpy()
            outputs = (outputs * 0.5 + 0.5).cpu().numpy()
            
            # Save first few reconstructed samples
            for i in range(min(5, len(inputs))):
                plt.figure(figsize=(4, 2))
                plt.subplot(1, 2, 1)
                plt.imshow(np.transpose(inputs[i], (1, 2, 0)))
                plt.title("Original")
                plt.subplot(1, 2, 2)
                plt.imshow(np.transpose(outputs[i], (1, 2, 0)))
                plt.title("Reconstructed")
                plt.savefig(f"{output_dir}/epoch_{epoch + 1}_sample_{i}.png")
                plt.close()

# Training Loop with Scheduler
def train_autoencoder(model, dataloader, optimizer, criterion, scaler, total_epochs, start_epoch=0):
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
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

        # Save reconstruction and checkpoint
        save_reconstruction_sample(model, dataloader, epoch, device)
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, loss_history, checkpoint_path)

        scheduler.step()  # Adjust learning rate

    return loss_history

# Main Execution
if __name__ == "__main__":
    autoencoder = Autoencoder(latent_dim).to(device)
    optimizer = optim.AdamW(autoencoder.parameters(), lr=lr, weight_decay=1e-5)
    perceptual_loss = PerceptualLoss().to(device)
    scaler = GradScaler()

    # Load checkpoint if available
    autoencoder, optimizer, start_epoch, loss_history = load_checkpoint(autoencoder, optimizer, checkpoint_path)

    # Train the model
    loss_history = train_autoencoder(autoencoder, dataloader, optimizer, perceptual_loss, scaler, epochs, start_epoch)
