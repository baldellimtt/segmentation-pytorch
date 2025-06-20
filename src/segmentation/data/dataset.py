import os
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for image segmentation tasks.
    Loads images and masks, applies augmentations and preprocessing.
    """
    def __init__(self, image_name, mask_name, transform=None, image_size=(512, 512)):
        """
        Initialize the dataset.
        Args:
            image_name (list): List of image file paths.
            mask_name (list): List of mask file paths.
            transform (callable, optional): A function/transform to apply to both image and mask.
            image_size (tuple): Size to resize images and masks to (width, height).
        """
        self.image_name = image_name
        self.mask_name = mask_name
        self.size = image_size
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.Resize(self.size),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.size),
            ]
        )
        self.add_transform = transform

    def __len__(self):
        """
        Return the number of samples in the dataset.
        Returns:
            int: Number of samples.
        """
        return len(self.mask_name)

    def __getitem__(self, idx):
        """
        Get a sample (image, mask) by index.
        Loads the image and mask, applies augmentations and preprocessing.
        Args:
            idx (int): Index of the sample.
        Returns:
            tuple: (image tensor, mask tensor)
        """
        # Get the image and mask file paths
        image_path = self.image_name[idx]
        mask_path = self.mask_name[idx]
        # Load the image and the mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 'L' for grayscale mask
        image = np.array(image)
        mask = np.array(mask)
        # Apply transformations
        if self.add_transform:
            # Apply geometric transforms to both image and mask
            augmented = self.add_transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        if self.image_transform:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)
        return image, mask
