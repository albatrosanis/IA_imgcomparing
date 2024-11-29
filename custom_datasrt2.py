import os
from torch.utils.data import Dataset
from PIL import Image

def to_rgb(img):
    """
    Converts an image to RGB format. Handles cases where the image might be in grayscale, RGBA, 
    or other modes.
    """
    if img.mode == 'RGBA':
        img = img.convert('RGB')  # Convert RGBA to RGB
    elif img.mode != 'RGB':
        img = img.convert('RGB')  # Convert grayscale or other modes to RGB
    return img

class CustomPokemonDataset(Dataset):
    """
    Custom dataset class for loading Pokemon images.
    - Expects images to be located in the `data_dir`.
    - Applies any transformations provided during initialization.
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing images.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        """
        Returns:
            int: Total number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            torch.Tensor: Transformed image.
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path)  # Open the image
        image = to_rgb(image)  # Ensure the image is in RGB format

        if self.transform:
            image = self.transform(image)  # Apply transformations

        return image
