"""
TrendLens — Preprocessing Module
PyTorch Dataset and DataLoader for Fashion Image Ingestion
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.preprocessing.image_loader import (
    load_and_validate_image,
    get_default_transform,
)


class FashionDataset(Dataset):
    """
    PyTorch Dataset for TrendLens Fashion images and metadata.
    Loads and transforms images lazily on request to optimize RAM usage.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        transform: Optional[Callable] = None,
        image_size: int = 224,
        root_dir: Optional[Union[str, Path]] = None
    ):
        """
        Args:
            metadata_df: DataFrame containing at least ['image_id', 'file_path'].
                         Optional columns: ['category', 'timestamp'].
            transform: Custom PyTorch transform pipeline. If None, default is used.
            image_size: Target image resolution (224x224).
            root_dir: Base directory to resolve relative file paths, if needed.
        """
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.image_size = image_size
        self.transform = transform or get_default_transform(image_size=image_size)
        self.root_dir = Path(root_dir) if root_dir else None

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, Dict]]:
        row = self.metadata_df.iloc[idx]
        image_id = str(row.get("image_id", idx))
        file_path = Path(row["file_path"])

        if self.root_dir and not file_path.is_absolute():
            file_path = self.root_dir / file_path

        # Load and validate image
        img = load_and_validate_image(file_path)

        if img is None:
            # Fallback for corrupted images: create a blank normalized tensor to avoid crashing
            # This is marked with a flag 'is_valid': False
            tensor = torch.zeros(3, self.image_size, self.image_size)
            is_valid = False
        else:
            tensor = self.transform(img)
            is_valid = True

        return {
            "image": tensor,
            "image_id": image_id,
            "file_path": str(file_path.as_posix()),
            "category": str(row.get("category", "unknown")),
            "timestamp": str(row.get("timestamp", "2024-01-01")),
            "is_valid": is_valid
        }


def create_fashion_dataloader(
    dataset: FashionDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """
    Creates a PyTorch DataLoader configured for batch processing.
    
    Args:
        dataset: Instance of FashionDataset.
        batch_size: Number of images per batch.
        shuffle: Whether to shuffle (False for feature extraction/inference).
        num_workers: Multi-process data loading workers (default 0 for Windows stability).
        
    Returns:
        torch.utils.data.DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
