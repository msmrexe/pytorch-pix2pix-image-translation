"""
Implements the Cityscapes Dataset class and data loading functions.
"""
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from src import config

class CityscapesDataset(Dataset):
    """
    Custom Dataset for loading Cityscapes Pix2Pix data.
    The dataset expects images where the left half is the real image
    and the right half is the segmented (label) image.
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Directory path containing the images.
            transform (callable, optional): Optional transform to be applied.
        """
        self.data_dir = data_dir
        self.transform = transform
        try:
            self.files = sorted(os.listdir(self.data_dir))
            if not self.files:
                raise FileNotFoundError(f"No files found in directory: {data_dir}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please ensure the dataset is downloaded and in the correct path.")
            print("Try running: bash setup_dataset.sh")
            self.files = []

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if not self.files:
            raise IndexError("Dataset is empty. Check data directory.")
            
        try:
            image_path = os.path.join(self.data_dir, self.files[idx])
            image = Image.open(image_path).convert("RGB")

            width, height = image.size
            # The dataset format is: Real Image (Left) | Segmented Image (Right)
            # We crop to separate them.
            real_image = image.crop((0, 0, width // 2, height))
            segmented_image = image.crop((width // 2, 0, width, height))

            if self.transform:
                # Apply the same transform to both
                real_image = self.transform(real_image)
                segmented_image = self.transform(segmented_image)
            
            # We return (Input, Target) -> (Segmented, Real)
            return segmented_image, real_image
            
        except Exception as e:
            print(f"Error loading image {self.files[idx]}: {e}")
            return None, None # Dataloader will skip this

def get_data_transforms(size):
    """Returns the torchvision transforms for the dataset."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(config.NORM_MEAN, config.NORM_STD), # Normalize to [-1, 1]
    ])

def get_loaders(train_dir, val_dir, batch_size, img_size):
    """
    Creates and returns the training and validation DataLoaders.
    """
    data_transforms = get_data_transforms(img_size)

    train_dataset = CityscapesDataset(train_dir, transform=data_transforms)
    val_dataset = CityscapesDataset(val_dir, transform=data_transforms)

    # Handle empty dataset case
    if not train_dataset or not val_dataset:
        return None, None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True
    )

    return train_loader, val_loader
