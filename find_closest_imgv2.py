import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
from tqdm import tqdm

# Define Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)
        loss = torch.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

# Dataset with anchor, positive, and negative samples
class TripletDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.image_paths = [os.path.join(dataset_dir, p) for p in os.listdir(dataset_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        anchor_path = self.image_paths[idx]
        anchor_image = Image.open(anchor_path).convert("RGB")
        positive_path = anchor_path  # For simplicity, use the same image as positive
        negative_path = self._get_random_negative(anchor_path)

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(Image.open(positive_path).convert("RGB"))
            negative_image = self.transform(Image.open(negative_path).convert("RGB"))

        return anchor_image, positive_image, negative_image

    def _get_random_negative(self, anchor_path):
        negative_path = anchor_path
        while negative_path == anchor_path:
            negative_path = self.image_paths[torch.randint(0, len(self.image_paths), (1,)).item()]
        return negative_path

# Data Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define the Encoder (Using a Pretrained ResNet Backbone)
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        backbone = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # Remove FC layer
        self.fc = nn.Linear(backbone.fc.in_features, latent_dim)

    def forward(self, x):
        features = self.feature_extractor(x).flatten(1)  # Flatten spatial dimensions
        latent = self.fc(features)
        return latent

# Training Loop
def train_model(encoder, dataloader, criterion, optimizer, device, epochs=10):
    encoder.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for anchors, positives, negatives in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)

            # Forward pass
            anchor_latents = encoder(anchors)
            positive_latents = encoder(positives)
            negative_latents = encoder(negatives)

            # Compute loss
            loss = criterion(anchor_latents, positive_latents, negative_latents)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(dataloader):.4f}")

# Initialize Dataset and DataLoader
dataset_dir = "C:/Users/Aures/Documents/GitHub/Train_Encode_Decode_IA/pokemon"
dataset = TripletDataset(dataset_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Hyperparameters and Model Initialization
latent_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder(latent_dim).to(device)
criterion = TripletLoss(margin=1.0)
optimizer = torch.optim.AdamW(encoder.parameters(), lr=0.001, weight_decay=1e-4)

# Train the Model
train_model(encoder, dataloader, criterion, optimizer, device, epochs=20)

# Save the Model
torch.save(encoder.state_dict(), "encoder_triplet.pth")

# Encode Image Function
def encode_image(image_path, encoder, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        latent_vector = encoder(image)
    return latent_vector

# Find Closest Image
def find_closest_image(input_image_path, dataset_dir, encoder, transform):
    input_latent = encode_image(input_image_path, encoder, transform)

    closest_image_path = None
    min_distance = float('inf')

    for file_name in tqdm(os.listdir(dataset_dir), desc="Searching dataset"):
        image_path = os.path.join(dataset_dir, file_name)
        dataset_latent = encode_image(image_path, encoder, transform)
        distance = torch.norm(input_latent - dataset_latent).item()

        if distance < min_distance:
            min_distance = distance
            closest_image_path = image_path

    return closest_image_path, min_distance

# Example Usage
if __name__ == "__main__":
    # Load the trained encoder
    encoder.load_state_dict(torch.load("encoder_triplet.pth"))
    encoder.eval()

    input_image_path = "006.png"
    closest_image_path, distance = find_closest_image(input_image_path, dataset_dir, encoder, transform)

    print(f"Closest match: {closest_image_path} with distance {distance:.4f}")
