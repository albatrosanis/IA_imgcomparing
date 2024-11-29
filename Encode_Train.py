import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
import multiprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Ensure correct multiprocessing context
multiprocessing.set_start_method('spawn', force=True)

# Parameters
batch_size = 32
lr = 1e-4
epochs = 50
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
from Custom_datasrt import CustomPokemonDataset  # Correct class name
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

# Recognition Model (Classifier Head)
class Recognizer(nn.Module):
    def __init__(self, encoder, num_classes):
        super(Recognizer, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        latent = self.encoder(x)
        return self.fc(latent)

# Initialize Model, Optimizer, and Loss
autoencoder = Autoencoder(latent_dim).to(device)
optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training Function
def train_autoencoder(model, dataloader, optimizer, criterion, epochs):
    loss_history = []
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}'):
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
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'autoencoder_epoch_{epoch+1}.pth')

    return loss_history

# Visualization Function
def visualize_reconstruction(model, dataloader, num_images=5):
    model.eval()
    with torch.no_grad():
        data_iter = iter(dataloader)
        images = next(data_iter).to(device)
        noisy_images = images + noise_scale * torch.randn_like(images)
        noisy_images = torch.clamp(noisy_images, -1., 1.)
        reconstructed = model(noisy_images)

        fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
        for i in range(num_images):
            axes[i, 0].imshow((images[i].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1))
            axes[i, 0].set_title("Original")
            axes[i, 1].imshow((noisy_images[i].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1))
            axes[i, 1].set_title("Noisy")
            axes[i, 2].imshow((reconstructed[i].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1))
            axes[i, 2].set_title("Reconstructed")
        plt.tight_layout()
        plt.show()

# Clustering for Recognition
def perform_clustering(model, dataloader, num_clusters):
    model.eval()
    latent_vectors = []
    with torch.no_grad():
        for data in dataloader:
            latent = model.encoder(data.to(device))
            latent_vectors.append(latent.cpu())
    latent_vectors = torch.cat(latent_vectors).numpy()
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(latent_vectors)
    score = silhouette_score(latent_vectors, labels)
    print(f"Clustering Silhouette Score: {score:.4f}")
    return labels

# Main Execution
if __name__ == "__main__":
    # Train Autoencoder
    loss_history = train_autoencoder(autoencoder, dataloader, optimizer, criterion, epochs)

    # Plot Loss
    plt.plot(loss_history, label="Reconstruction Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Save Final Model
    torch.save(autoencoder.state_dict(), 'autoencoder_final.pth')

    # Visualize Reconstruction
    visualize_reconstruction(autoencoder, dataloader)

    # Perform Clustering (Recognition)
    labels = perform_clustering(autoencoder, dataloader, num_clusters=10)
