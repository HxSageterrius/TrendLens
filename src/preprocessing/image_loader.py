"""
TrendLens — Preprocessing Module
Image Loading, Validation, and Transformation Pipeline
"""

import logging
from pathlib import Path
from typing import Optional, Union, Tuple
from PIL import Image, ImageOps
import torch
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_and_validate_image(image_path: Union[str, Path]) -> Optional[Image.Image]:
    """
    Safely load an image from disk, verify its integrity, and convert it to 3-channel RGB.
    
    Handles:
    - Missing files
    - Corrupted or truncated files
    - Grayscale (1-channel), RGBA (4-channel), CMYK profiles
    - EXIF orientation correction
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        PIL Image in RGB mode if valid, or None if corrupted / missing.
    """
    path = Path(image_path)
    if not path.is_file():
        logger.warning(f"Image file does not exist: {path}")
        return None

    try:
        with Image.open(path) as img:
            # Force load the image data to catch truncation or byte corruption
            img.load()
            
            # Auto-orient based on EXIF tag (avoids rotated fashion photos)
            img = ImageOps.exif_transpose(img)
            
            # Convert to standard 3-channel RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            return img.copy()
            
    except Exception as e:
        logger.error(f"Failed to load or validate image '{path}': {e}")
        return None


def get_default_transform(
    image_size: int = 224,
    preserve_aspect_ratio: bool = True
) -> transforms.Compose:
    """
    Constructs a PyTorch torchvision transformation pipeline.
    Standardized for deep learning feature backbones (ResNet, CLIP, etc.).
    
    Args:
        image_size: Target dimension (default 224x224).
        preserve_aspect_ratio: If True, resizes shorter edge and center crops.
                               If False, directly resizes (may stretch).
                               
    Returns:
        torchvision.transforms.Compose pipeline.
    """
    transform_list = []
    
    if preserve_aspect_ratio:
        # Resize smaller dimension to image_size, then center crop
        transform_list.extend([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size)
        ])
    else:
        # Direct resize to (image_size, image_size)
        transform_list.append(
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR)
        )

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return transforms.Compose(transform_list)


def preprocess_image(
    image_path: Union[str, Path],
    transform: Optional[transforms.Compose] = None,
    image_size: int = 224
) -> Optional[torch.Tensor]:
    """
    Convenience function: Loads, validates, and transforms a single image into a PyTorch tensor.
    
    Args:
        image_path: Path to the image file.
        transform: Optional custom transform. If None, uses default transform.
        image_size: Target size if default transform is generated.
        
    Returns:
        torch.Tensor of shape (3, image_size, image_size) or None if invalid.
    """
    img = load_and_validate_image(image_path)
    if img is None:
        return None
        
    if transform is None:
        transform = get_default_transform(image_size=image_size)
        
    return transform(img)
