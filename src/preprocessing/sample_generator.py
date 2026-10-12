"""
TrendLens — Preprocessing Module
Synthetic Fashion Sample Dataset Generator for Prototyping & Pipeline Testing
"""

from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.preprocessing.metadata import assign_synthetic_timeline


CATEGORIES_PALETTE: Dict[str, tuple] = {
    "dress": (230, 80, 110),      # Rose / Pink
    "shirt": (70, 130, 180),      # Steel Blue
    "trousers": (60, 60, 80),     # Dark Slate
    "jacket": (160, 82, 45),      # Brown / Leather
}


def generate_synthetic_fashion_dataset(
    output_dir: Path,
    num_samples_per_category: int = 5,
    image_size: tuple = (256, 256)
) -> pd.DataFrame:
    """
    Generates synthetic fashion-like image samples with distinct color palettes
    and geometry, along with an associated metadata CSV.
    
    This ensures that TrendLens can be developed, tested, and demonstrated
    immediately without waiting for multi-gigabyte dataset downloads.
    
    Args:
        output_dir: Directory where samples and metadata.csv will be saved.
        num_samples_per_category: Number of images to generate for each category.
        image_size: (width, height) of generated images.
        
    Returns:
        DataFrame with image metadata and timestamps.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    sample_counter = 1

    for category, base_color in CATEGORIES_PALETTE.items():
        for i in range(num_samples_per_category):
            # Introduce subtle color variance within the category
            noise = np.random.randint(-20, 20, size=3)
            color = tuple(np.clip(np.array(base_color) + noise, 0, 255).astype(int))

            # Create base canvas (light background)
            img = Image.new("RGB", image_size, color=(245, 245, 245))
            draw = ImageDraw.Draw(img)

            w, h = image_size
            # Draw distinctive geometric garments
            if category == "dress":
                # A-line silhouette
                draw.polygon([(w*0.35, h*0.2), (w*0.65, h*0.2), (w*0.8, h*0.85), (w*0.2, h*0.85)], fill=color)
            elif category == "shirt":
                # Torso box with sleeves
                draw.rectangle([(w*0.25, h*0.25), (w*0.75, h*0.75)], fill=color)
                draw.polygon([(w*0.25, h*0.25), (w*0.1, h*0.45), (w*0.25, h*0.5)], fill=color)
                draw.polygon([(w*0.75, h*0.25), (w*0.9, h*0.45), (w*0.75, h*0.5)], fill=color)
            elif category == "trousers":
                # Two vertical legs
                draw.rectangle([(w*0.3, h*0.2), (w*0.7, h*0.4)], fill=color)
                draw.rectangle([(w*0.3, h*0.4), (w*0.47, h*0.88)], fill=color)
                draw.rectangle([(w*0.53, h*0.4), (w*0.7, h*0.88)], fill=color)
            elif category == "jacket":
                # Heavier coat shape with collar
                draw.rectangle([(w*0.2, h*0.2), (w*0.8, h*0.8)], fill=color)
                draw.polygon([(w*0.4, h*0.2), (w*0.5, h*0.4), (w*0.6, h*0.2)], fill=(200, 200, 200))

            file_name = f"sample_{sample_counter:04d}_{category}.jpg"
            file_path = images_dir / file_name
            img.save(file_path, quality=95)

            records.append({
                "image_id": f"SMPL_{sample_counter:04d}",
                "file_path": str(file_path.relative_to(output_dir.parent.parent).as_posix()),
                "category": category
            })
            sample_counter += 1

    df = pd.DataFrame(records)
    # Assign chronological dates from 2023 to 2024 for time-series forecasting
    df_dated = assign_synthetic_timeline(df, start_date="2023-01-01", end_date="2024-12-31")

    csv_path = output_dir / "samples_metadata.csv"
    df_dated.to_csv(csv_path, index=False)
    return df_dated


if __name__ == "__main__":
    target = Path("data/samples")
    df = generate_synthetic_fashion_dataset(target, num_samples_per_category=5)
    print(f"Generated {len(df)} sample fashion images in {target}")
