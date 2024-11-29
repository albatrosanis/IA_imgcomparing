import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from Custom_datasrt import CustomPokemonDataset
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Load the pre-trained autoencoder
class Autoencoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(512 * 8 * 8, latent_dim),
            torch.nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

# Helper function to find the most similar image
def find_most_similar_image(query_embedding, embeddings, dataset):
    # Compute cosine similarity
    similarity = cosine_similarity(query_embedding.reshape(1, -1), embeddings)
    most_similar_idx = np.argmax(similarity)

    # Retrieve the most similar image from the dataset
    similar_image_tensor = dataset[most_similar_idx]
    if isinstance(similar_image_tensor, tuple):  # Handle dataset returning tuples
        similar_image_tensor = similar_image_tensor[0]

    if similar_image_tensor.ndim == 3:  # Expected shape: (C, H, W)
        similar_image = similar_image_tensor.permute(1, 2, 0).numpy()  # Convert to HWC for visualization
    else:
        raise ValueError(f"Expected 3D tensor, got {similar_image_tensor.ndim}D tensor instead.")

    similarity_score = similarity[0, most_similar_idx]
    return similar_image, similarity_score

# Main execution
if __name__ == "__main__":
    latent_dim = 128
    device = torch.device("cpu")

    # Load autoencoder and embeddings
    autoencoder = Autoencoder(latent_dim).to(device)
    autoencoder.load_state_dict(torch.load("autoencoder_final.pth", map_location=device))
    autoencoder.eval()

    embeddings = np.load("embeddings.npy")  # Precomputed embeddings

    # Initialize dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = CustomPokemonDataset(data_dir="path_to_pokemon_images", transform=transform)

    # Load and process query image
    query_image_path = "path_to_query_image"
    query_image = Image.open(query_image_path).convert("RGB")
    query_tensor = transform(query_image).unsqueeze(0).to(device)

    # Generate embedding for the query image
    with torch.no_grad():
        query_embedding = autoencoder.encoder(query_tensor).cpu().numpy()

    # Find the most similar image
    similar_image, similarity_score = find_most_similar_image(query_embedding, embeddings, dataset)

    # Display query image and the most similar image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Query Image")
    plt.imshow(query_image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Most Similar Image\n(Similarity: {similarity_score:.2f})")
    plt.imshow(similar_image)
    plt.axis("off")

    plt.show()
