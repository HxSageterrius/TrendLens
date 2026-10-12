"""
TrendLens — Unit Tests for Module 1 (Dataset Ingestion & Preprocessing)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from src.preprocessing.image_loader import (
    load_and_validate_image,
    get_default_transform,
    preprocess_image,
)
from src.preprocessing.metadata import (
    DEEPFASHION2_CATEGORIES,
    create_metadata_record,
    assign_synthetic_timeline,
    parse_deepfashion2_annotation,
)
from src.preprocessing.dataset import (
    FashionDataset,
    create_fashion_dataloader,
)


@pytest.fixture
def temp_image_dir(tmp_path):
    """Fixture to create sample test images in a temporary directory."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # 1. Valid standard RGB image (300x400)
    rgb_img = Image.new("RGB", (300, 400), color=(200, 50, 50))
    rgb_path = img_dir / "sample_rgb.jpg"
    rgb_img.save(rgb_path)

    # 2. RGBA image with alpha transparency (200x200)
    rgba_img = Image.new("RGBA", (200, 200), color=(50, 200, 50, 128))
    rgba_path = img_dir / "sample_rgba.png"
    rgba_img.save(rgba_path)

    # 3. Grayscale image (1 channel, 150x150)
    gray_img = Image.new("L", (150, 150), color=120)
    gray_path = img_dir / "sample_gray.jpg"
    gray_img.save(gray_path)

    # 4. Corrupted / broken image (random raw bytes)
    corrupt_path = img_dir / "sample_corrupt.jpg"
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_A_VALID_JPEG_HEADER_RANDOM_BYTES_XYZ")

    return {
        "dir": img_dir,
        "rgb": rgb_path,
        "rgba": rgba_path,
        "gray": gray_path,
        "corrupt": corrupt_path
    }


def test_load_and_validate_image_valid_rgb(temp_image_dir):
    """Verifies that a valid RGB image loads correctly."""
    img = load_and_validate_image(temp_image_dir["rgb"])
    assert img is not None
    assert img.mode == "RGB"
    assert img.size == (300, 400)


def test_load_and_validate_image_rgba_and_gray(temp_image_dir):
    """Verifies that RGBA and Grayscale images are converted to 3-channel RGB."""
    img_rgba = load_and_validate_image(temp_image_dir["rgba"])
    assert img_rgba is not None
    assert img_rgba.mode == "RGB"

    img_gray = load_and_validate_image(temp_image_dir["gray"])
    assert img_gray is not None
    assert img_gray.mode == "RGB"


def test_load_and_validate_corrupted_image(temp_image_dir):
    """Verifies that corrupted files return None instead of raising unhandled exceptions."""
    img = load_and_validate_image(temp_image_dir["corrupt"])
    assert img is None


def test_preprocess_image_tensor_shape(temp_image_dir):
    """Verifies that preprocessing returns a tensor of exact dimensions (3, 224, 224)."""
    tensor = preprocess_image(temp_image_dir["rgb"], image_size=224)
    assert tensor is not None
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_metadata_record_and_timeline():
    """Verifies metadata record creation and chronological timeline assignment."""
    records = [
        create_metadata_record("img_1", "path/to/img1.jpg", "dress"),
        create_metadata_record("img_2", "path/to/img2.jpg", "trousers"),
        create_metadata_record("img_3", "path/to/img3.jpg", "shirt"),
    ]
    df = pd.DataFrame(records)
    assert len(df) == 3
    assert "timestamp" in df.columns

    # Test timeline assignment
    dated_df = assign_synthetic_timeline(df, start_date="2024-01-01", end_date="2024-06-30")
    assert len(dated_df) == 3
    # Check that dates are valid strings
    for ts in dated_df["timestamp"]:
        assert len(ts) == 10  # YYYY-MM-DD


def test_fashion_dataset_and_dataloader(temp_image_dir):
    """Verifies PyTorch Dataset and DataLoader batch loading."""
    df = pd.DataFrame([
        {"image_id": "1", "file_path": str(temp_image_dir["rgb"]), "category": "dress", "timestamp": "2024-01-01"},
        {"image_id": "2", "file_path": str(temp_image_dir["rgba"]), "category": "skirt", "timestamp": "2024-01-02"},
        {"image_id": "3", "file_path": str(temp_image_dir["corrupt"]), "category": "shirt", "timestamp": "2024-01-03"},
    ])

    dataset = FashionDataset(df, image_size=224)
    assert len(dataset) == 3

    # Check item access
    item_0 = dataset[0]
    assert item_0["is_valid"] is True
    assert item_0["image"].shape == (3, 224, 224)
    assert item_0["category"] == "dress"

    # Check corrupted item graceful handling
    item_corrupt = dataset[2]
    assert item_corrupt["is_valid"] is False
    assert item_corrupt["image"].shape == (3, 224, 224)

    # Check DataLoader batching
    loader = create_fashion_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["image"].shape == (2, 3, 224, 224)
    assert len(batch["image_id"]) == 2
