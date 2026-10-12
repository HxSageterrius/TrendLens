"""
TrendLens — Preprocessing Module
Metadata Handler, DeepFashion2 Label Parser, and Temporal Alignment
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard DeepFashion2 category definitions (13 primary classes)
DEEPFASHION2_CATEGORIES: Dict[int, str] = {
    1: "short_sleeved_shirt",
    2: "long_sleeved_shirt",
    3: "short_sleeved_outwear",
    4: "long_sleeved_outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short_sleeved_dress",
    11: "long_sleeved_dress",
    12: "vest_dress",
    13: "sling_dress"
}


def parse_deepfashion2_annotation(annotation_path: Union[str, Path]) -> List[Dict]:
    """
    Parses a single DeepFashion2 JSON annotation file.
    DeepFashion2 annotations contain item-level bounding boxes and category_id.
    
    Args:
        annotation_path: Path to the annotation JSON file.
        
    Returns:
        List of detected item records with category_id and category_name.
    """
    path = Path(annotation_path)
    if not path.is_file():
        logger.warning(f"Annotation file not found: {path}")
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        items = []
        for key, val in data.items():
            if key.startswith("item"):
                cat_id = val.get("category_id")
                items.append({
                    "item_key": key,
                    "category_id": cat_id,
                    "category_name": DEEPFASHION2_CATEGORIES.get(cat_id, "unknown"),
                    "bounding_box": val.get("bounding_box", []),
                    "style": val.get("style", 0)
                })
        return items
    except Exception as e:
        logger.error(f"Error parsing annotation file '{path}': {e}")
        return []


def create_metadata_record(
    image_id: str,
    file_path: Union[str, Path],
    category: str = "general_fashion",
    timestamp: Optional[str] = None
) -> Dict[str, Union[str, Path]]:
    """
    Creates a single standardized metadata dictionary.
    
    Args:
        image_id: Unique string identifier for the image.
        file_path: Relative or absolute path to the image file.
        category: Fashion category name.
        timestamp: Date string in 'YYYY-MM-DD' format.
        
    Returns:
        Dictionary adhering to TrendLens metadata schema.
    """
    return {
        "image_id": str(image_id),
        "file_path": str(Path(file_path).as_posix()),
        "category": str(category),
        "timestamp": timestamp or "2024-01-01"
    }


def assign_synthetic_timeline(
    df: pd.DataFrame,
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42
) -> pd.DataFrame:
    """
    Assigns realistic, chronological timestamps to an image collection.
    
    Context / Rationale:
    DeepFashion2 images do not include native timestamps. For TrendLens to forecast
    trends over time using Prophet, each product image needs a valid timestamp.
    This function assigns a uniformly sampled or seasonal date for reproducible testing.
    
    Args:
        df: DataFrame containing at least ['image_id', 'file_path', 'category'].
        start_date: Start date for the timeline ('YYYY-MM-DD').
        end_date: End date for the timeline ('YYYY-MM-DD').
        seed: Random seed for reproducibility.
        
    Returns:
        DataFrame with an updated, validated 'timestamp' column.
    """
    if df.empty:
        return df

    np.random.seed(seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    
    df_copy = df.copy()
    random_dates = np.random.choice(date_range, size=len(df_copy))
    df_copy["timestamp"] = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in random_dates]
    
    # Sort chronologically
    df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])
    df_copy = df_copy.sort_values("timestamp").reset_index(drop=True)
    df_copy["timestamp"] = df_copy["timestamp"].dt.strftime("%Y-%m-%d")
    
    return df_copy
