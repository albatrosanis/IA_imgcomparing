import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CustomPokemonDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_name = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_name)

        # Apply transformations (resize, normalization, etc.)
        if self.transform:
            image = self.transform(image)

        # Return the image tensor
        return image
