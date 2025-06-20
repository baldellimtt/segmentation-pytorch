import os
import glob

import pytorch_lightning as pl
import albumentations as albu
from torch.utils.data import DataLoader
from pathlib import Path

from src.segmentation.data.dataset import SegmentationDataset


class SegmentationDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for segmentation tasks.
    Loads data from train/val/test folders, each containing images/ and masks/ subfolders.
    Handles data loading, augmentation, and dataloader creation for each split.
    """
    def __init__(self, config):
        """
        Initialize the DataModule from a configuration dictionary.
        Args:
            config (dict): Configuration dictionary containing:
                - data.splits_dir: Path to the folder containing split folders (train/val/test)
                - training.batch_size: Batch size for dataloaders
                - model.input_size: (height, width) for resizing images and masks
                - augmentation: List of augmentation configs (optional)
        """
        super().__init__()
        # Make splits_dir absolute if not already
        splits_dir = config['data']['splits_dir']
        if not os.path.isabs(splits_dir):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            splits_dir = os.path.abspath(os.path.join(project_root, splits_dir))
        self.splits_dir = Path(splits_dir)
        self.batch_size = config['training']['batch_size']
        self.resize_size = tuple(config['model']['input_size'])
        self.augmentation_config = config.get('augmentation', None)
        self.num_workers = config['training'].get('num_workers', 2)

    def get_training_augmentation(self):
        """
        Build the training augmentation pipeline using albumentations.
        If augmentation_config is provided, builds dynamically from config.
        Otherwise, uses a default set of augmentations.
        Returns:
            albumentations.Compose: The composed augmentation pipeline.
        """
        def build_transform(transform_cfg):
            name = transform_cfg['name']
            if name == 'OneOf':
                transforms = [build_transform(t) for t in transform_cfg['transforms']]
                p = transform_cfg.get('p', 0.5)
                return albu.OneOf(transforms, p=p)
            elif name == 'Compose':
                transforms = [build_transform(t) for t in transform_cfg['transforms']]
                p = transform_cfg.get('p', 1.0)
                return albu.Compose(transforms, p=p)
            else:
                cls = getattr(albu, name)
                params = {k: v for k, v in transform_cfg.items() if k != 'name' and k != 'transforms'}
                return cls(**params)

        if self.augmentation_config is None:
            geometric_transforms = [albu.HorizontalFlip(p=0.9)]
            image_only_transforms = [
                albu.GaussNoise(p=0.3),
                albu.OneOf([
                    albu.RandomBrightnessContrast(p=0.6),
                    albu.RandomGamma(p=0.4),
                ], p=0.8),
                albu.Blur(p=0.5),
                albu.GridDropout(ratio=0.3, unit_size_range=(10, 20), fill="random_uniform", p=0.2),
                albu.RandomFog(alpha=0.03, p=0.2),
            ]
            train_transforms = geometric_transforms + image_only_transforms
            return albu.Compose(train_transforms, additional_targets={'mask': 'mask'})
        else:
            transforms_list = [build_transform(t) for t in self.augmentation_config]
            return albu.Compose(transforms_list, additional_targets={'mask': 'mask'})

    def load_split_filenames(self, split_folder):
        """
        Load image and mask file paths from a split folder (images/ and masks/).
        Args:
            split_folder (str): Name of the split folder (e.g., 'train', 'val', 'test').
        Returns:
            tuple: (list of image paths, list of mask paths)
        """
        images_path = os.path.join(self.splits_dir, split_folder, 'images')
        masks_path = os.path.join(self.splits_dir, split_folder, 'masks')
        image_files = sorted(glob.glob(os.path.join(images_path, '*.png')))
        mask_files = [os.path.join(masks_path, os.path.basename(f)) for f in image_files]
        for m in mask_files:
            if not os.path.exists(m):
                raise FileNotFoundError(f"Mask not found: {m}")
        return image_files, mask_files

    def setup(self, stage=None):
        """
        Setup method to split dataset into train, val, and test.
        Initializes SegmentationDataset objects for each split.
        Args:
            stage (str or None): Stage to set up (unused, for Lightning compatibility).
        """
        train_paths, train_labels = self.load_split_filenames("train")
        val_paths, val_labels = self.load_split_filenames("val")
        test_paths, test_labels = self.load_split_filenames("test")
        self.train_dataset = SegmentationDataset(
            train_paths, train_labels, self.get_training_augmentation(), self.resize_size
        )
        self.val_dataset = SegmentationDataset(val_paths, val_labels, None, self.resize_size)
        self.test_dataset = SegmentationDataset(test_paths, test_labels, None, self.resize_size)

    def train_dataloader(self):
        """
        Return the train dataloader.
        Returns:
            torch.utils.data.DataLoader: Dataloader for the training set.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """
        Return the validation dataloader.
        Returns:
            torch.utils.data.DataLoader: Dataloader for the validation set.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        """
        Return the test dataloader.
        Returns:
            torch.utils.data.DataLoader: Dataloader for the test set.
        """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
