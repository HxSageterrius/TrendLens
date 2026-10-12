"""
TrendLens — Module 2: Visual Feature Extraction
Extracts deep visual embeddings from fashion images using PyTorch & ResNet/CNN architectures.
Generates L2-normalized feature vectors for clustering and similarity retrieval.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

from src.preprocessing.image_loader import load_and_validate_image, get_default_transform


class FeatureExtractor:
    """
    Deep Convolutional Feature Extractor using a pretrained vision backbone.
    Removes the classification head to extract dense visual representations.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        device: Optional[str] = None
    ):
        """
        Args:
            backbone_name: Architecture name ('resnet18' or 'resnet50').
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.backbone_name = backbone_name
        self.transform = get_default_transform(image_size=224)
        self.model = self._load_backbone(backbone_name)
        self.model.eval()
        self.model.to(self.device)

    def _load_backbone(self, name: str) -> nn.Module:
        """Loads vision model with classification head replaced by Identity."""
        if name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
            self.embedding_dim = model.fc.in_features
            model.fc = nn.Identity()
        else:
            # Default to fast, lightweight ResNet18
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            self.embedding_dim = model.fc.in_features
            model.fc = nn.Identity()

        return model

    @torch.no_grad()
    def extract_from_image(self, image: Union[Image.Image, str, Path]) -> np.ndarray:
        """
        Extracts an L2-normalized embedding vector for a single image.

        Args:
            image: PIL Image object or file path.

        Returns:
            1D numpy array of shape (embedding_dim,) normalized to unit length.
        """
        if isinstance(image, (str, Path)):
            pil_img = load_and_validate_image(image)
            if pil_img is None:
                raise ValueError(f"Unable to load or validate image: {image}")
        else:
            pil_img = image.convert("RGB")

        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        features = self.model(tensor)
        features = features.squeeze(0).cpu().numpy()

        # L2 Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features

    @torch.no_grad()
    def extract_from_catalog(
        self,
        metadata_df: pd.DataFrame,
        project_root: Optional[Path] = None,
        batch_size: int = 16
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Extracts embeddings for an entire catalog DataFrame.

        Args:
            metadata_df: DataFrame containing 'file_path' column.
            project_root: Root directory to resolve relative file_paths.
            batch_size: Batch size for inference.

        Returns:
            (embeddings_matrix, valid_metadata_df)
        """
        embeddings_list: List[np.ndarray] = []
        valid_indices: List[int] = []

        if project_root is None:
            project_root = Path.cwd()

        for idx, row in metadata_df.iterrows():
            rel_path = row["file_path"]
            full_path = project_root / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)

            if not full_path.is_file():
                continue

            try:
                emb = self.extract_from_image(full_path)
                embeddings_list.append(emb)
                valid_indices.append(idx)
            except Exception as e:
                continue

        if not embeddings_list:
            raise RuntimeError("No valid images could be processed for embeddings.")

        embeddings_matrix = np.vstack(embeddings_list)
        valid_df = metadata_df.loc[valid_indices].reset_index(drop=True)

        return embeddings_matrix, valid_df


def save_embeddings(
    embeddings: np.ndarray,
    metadata_df: pd.DataFrame,
    output_dir: Path,
    filename_prefix: str = "fashion_catalog"
):
    """Saves embeddings matrix (.npy) and synchronized metadata (.csv)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    emb_path = output_dir / f"{filename_prefix}_embeddings.npy"
    meta_path = output_dir / f"{filename_prefix}_metadata.csv"

    np.save(emb_path, embeddings)
    metadata_df.to_csv(meta_path, index=False)
    return emb_path, meta_path


def load_embeddings(
    output_dir: Path,
    filename_prefix: str = "fashion_catalog"
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Loads saved embeddings (.npy) and corresponding metadata (.csv)."""
    output_dir = Path(output_dir)
    emb_path = output_dir / f"{filename_prefix}_embeddings.npy"
    meta_path = output_dir / f"{filename_prefix}_metadata.csv"

    embeddings = np.load(emb_path)
    metadata_df = pd.read_csv(meta_path)
    return embeddings, metadata_df
