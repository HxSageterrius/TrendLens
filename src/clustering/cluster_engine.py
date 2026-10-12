"""
TrendLens — Module 3: Visual Clustering Pipeline
Applies K-Means and DBSCAN algorithms to visual embeddings.
Calculates cluster evaluation metrics (Silhouette Score), finds cluster exemplars,
and maps clusters to fashion trend styles.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


class VisualClusterEngine:
    """
    Engine for grouping visual embeddings into style clusters
    and analyzing cluster characteristics.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        random_state: int = 42
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model: Optional[KMeans] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.silhouette: float = 0.0

    def fit_kmeans(
        self,
        embeddings: np.ndarray,
        find_optimal_k: bool = False,
        k_range: Tuple[int, int] = (3, 8)
    ) -> Tuple[np.ndarray, float]:
        """
        Fits K-Means on the embeddings matrix.
        Optionally searches for optimal K using Silhouette Score.

        Args:
            embeddings: (N, D) array of feature vectors.
            find_optimal_k: Whether to sweep across k_range for best silhouette.
            k_range: (min_k, max_k) search bounds.

        Returns:
            (cluster_labels, silhouette_score)
        """
        n_samples = len(embeddings)
        if n_samples < 4:
            raise ValueError("Need at least 4 samples to perform meaningful clustering.")

        if find_optimal_k:
            best_k = self.n_clusters
            best_score = -1.0
            min_k, max_k = k_range
            max_k = min(max_k, n_samples - 1)

            for k in range(min_k, max_k + 1):
                km = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
                labels = km.fit_predict(embeddings)
                if len(set(labels)) > 1:
                    score = float(silhouette_score(embeddings, labels))
                    if score > best_score:
                        best_score = score
                        best_k = k

            self.n_clusters = best_k

        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto"
        )
        self.cluster_labels = self.model.fit_predict(embeddings)

        if len(set(self.cluster_labels)) > 1:
            self.silhouette = float(silhouette_score(embeddings, self.cluster_labels))
        else:
            self.silhouette = 0.0

        return self.cluster_labels, self.silhouette

    def fit_dbscan(
        self,
        embeddings: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 2
    ) -> Tuple[np.ndarray, float]:
        """
        Fits DBSCAN density-based clustering on embeddings.
        """
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        self.cluster_labels = db.fit_predict(embeddings)

        valid_mask = self.cluster_labels != -1
        if np.sum(valid_mask) > 2 and len(set(self.cluster_labels[valid_mask])) > 1:
            self.silhouette = float(silhouette_score(embeddings[valid_mask], self.cluster_labels[valid_mask]))
        else:
            self.silhouette = 0.0

        return self.cluster_labels, self.silhouette

    def get_cluster_summary(
        self,
        embeddings: np.ndarray,
        metadata_df: pd.DataFrame
    ) -> List[Dict]:
        """
        Produces detailed metadata for each cluster:
        - Cluster ID and size
        - Dominant style and category
        - Exemplar item (closest item to cluster centroid)
        - Silhouette metric
        """
        if self.cluster_labels is None:
            raise RuntimeError("Cluster engine has not been fitted yet.")

        df = metadata_df.copy()
        df["cluster_id"] = self.cluster_labels

        clusters_info = []
        unique_labels = sorted(list(set(self.cluster_labels)))

        for c_id in unique_labels:
            if c_id == -1:
                # Outlier noise in DBSCAN
                name = "Outliers / Noise"
            else:
                name = f"Cluster #{c_id + 1}"

            cluster_mask = (self.cluster_labels == c_id)
            c_items = df[cluster_mask]
            c_embeddings = embeddings[cluster_mask]
            count = len(c_items)

            # Find exemplar item (closest to mean embedding)
            centroid = np.mean(c_embeddings, axis=0, keepdims=True)
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
            sims = cosine_similarity(c_embeddings, centroid_norm).flatten()
            exemplar_idx = int(np.argmax(sims))
            exemplar_row = c_items.iloc[exemplar_idx].to_dict()

            # Dominant styles and categories
            top_styles = c_items["style"].value_counts().head(2).to_dict() if "style" in c_items else {}
            top_categories = c_items["category"].value_counts().head(2).to_dict() if "category" in c_items else {}

            clusters_info.append({
                "cluster_id": int(c_id),
                "cluster_name": name,
                "item_count": int(count),
                "exemplar_item": exemplar_row,
                "dominant_styles": top_styles,
                "dominant_categories": top_categories,
                "items": c_items.to_dict(orient="records")
            })

        return clusters_info

    def save(self, filepath: Path):
        """Saves fitted cluster engine model."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "VisualClusterEngine":
        """Loads fitted cluster engine."""
        return joblib.load(filepath)
