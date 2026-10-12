"""
TrendLens — Module 5: Visual Similarity Matching Engine
Computes cosine similarity between image feature vectors to perform Top-K visual search
and fashion recommendations based on user style queries.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from src.embeddings.extractor import FeatureExtractor


class SimilarityEngine:
    """
    Core Visual Similarity Engine for image retrieval and style matching.
    """

    def __init__(
        self,
        embeddings: Optional[np.ndarray] = None,
        metadata_df: Optional[pd.DataFrame] = None,
        feature_extractor: Optional[FeatureExtractor] = None
    ):
        self.embeddings = embeddings
        self.metadata_df = metadata_df
        self.feature_extractor = feature_extractor or FeatureExtractor()

    def set_catalog(self, embeddings: np.ndarray, metadata_df: pd.DataFrame):
        """Sets the active catalog embeddings and synchronized metadata."""
        if len(embeddings) != len(metadata_df):
            raise ValueError(
                f"Mismatch: embeddings length ({len(embeddings)}) != metadata rows ({len(metadata_df)})"
            )
        self.embeddings = embeddings
        self.metadata_df = metadata_df.reset_index(drop=True)

    def find_similar_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 6,
        category_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Finds Top-K most similar items in the catalog for a given embedding vector.

        Args:
            query_vector: 1D array of shape (embedding_dim,).
            top_k: Number of matches to return.
            category_filter: Optional category restriction.

        Returns:
            List of matching item dictionaries with similarity scores.
        """
        if self.embeddings is None or self.metadata_df is None:
            raise RuntimeError("Catalog embeddings or metadata have not been loaded.")

        # Ensure 2D query shape
        if query_vector.ndim == 1:
            q_norm = np.linalg.norm(query_vector)
            q_vec = (query_vector / (q_norm + 1e-9)).reshape(1, -1)
        else:
            q_vec = query_vector

        # Compute cosine similarity
        sims = cosine_similarity(q_vec, self.embeddings).flatten()

        df = self.metadata_df.copy()
        df["similarity_score"] = sims
        df["match_percentage"] = (sims * 100.0).round(1)

        if category_filter:
            df = df[df["category"].str.lower() == category_filter.lower()]

        # Sort descending
        top_df = df.sort_values("similarity_score", ascending=False).head(top_k)

        results = []
        for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
            item_dict = row.to_dict()
            item_dict["rank"] = rank
            results.append(item_dict)

        return results

    def find_similar_by_image(
        self,
        image_input: Union[Image.Image, str, Path],
        top_k: int = 6,
        category_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Finds Top-K similar items for a given PIL image or image file path.
        """
        query_vector = self.feature_extractor.extract_from_image(image_input)
        return self.find_similar_by_vector(
            query_vector=query_vector,
            top_k=top_k,
            category_filter=category_filter
        )

    def find_similar_by_item_id(
        self,
        item_id: str,
        top_k: int = 6
    ) -> List[Dict]:
        """
        Finds visually similar items to an existing catalog item ID (excluding itself).
        """
        if self.metadata_df is None:
            raise RuntimeError("Catalog metadata not set.")

        matches = self.metadata_df[self.metadata_df["item_id"] == item_id]
        if matches.empty:
            raise ValueError(f"Item ID '{item_id}' not found in catalog.")

        idx = matches.index[0]
        query_vector = self.embeddings[idx]

        results = self.find_similar_by_vector(query_vector=query_vector, top_k=top_k + 1)
        # Filter out self
        filtered = [r for r in results if r["item_id"] != item_id][:top_k]
        return filtered
