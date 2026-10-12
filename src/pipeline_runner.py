"""
TrendLens — Core Pipeline Orchestrator & Model Runner
Executes the full end-to-end ML pipeline:
1. Ingests fashion post metadata and images
2. Extracts deep visual feature embeddings (ResNet)
3. Fits visual clusters (K-Means) and computes silhouette evaluation
4. Generates time-series trend projections (Prophet / Statistical engine)
5. Validates similarity matching engine
"""

import json
from pathlib import Path
import sys

# Ensure UTF-8 output encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.embeddings.extractor import FeatureExtractor, save_embeddings
from src.clustering.cluster_engine import VisualClusterEngine
from src.forecasting.trend_forecaster import TrendForecaster
from src.similarity.similarity_engine import SimilarityEngine


def run_full_pipeline(project_root: Path) -> dict:
    print("=" * 60)
    print(">> Running TrendLens Core AI Pipeline")
    print("=" * 60)

    # Paths
    trends_dir = project_root / "data" / "fashion_trends"
    meta_csv = trends_dir / "fashion_posts_metadata.csv"
    time_series_csv = trends_dir / "insta_fashion_trends.csv"
    embeddings_dir = project_root / "data" / "embeddings"
    models_dir = project_root / "models"
    forecasts_dir = project_root / "data" / "forecasts"

    embeddings_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    forecasts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Catalog Metadata
    print("\n[Step 1] Loading fashion catalog metadata...")
    if not meta_csv.is_file():
        raise FileNotFoundError(f"Missing fashion metadata at: {meta_csv}")
    catalog_df = pd.read_csv(meta_csv)
    print(f"Loaded {len(catalog_df)} catalog items.")

    # 2. Extract Deep Visual Embeddings
    print("\n[Step 2] Extracting visual embeddings (ResNet)...")
    extractor = FeatureExtractor(backbone_name="resnet18")
    embeddings, valid_df = extractor.extract_from_catalog(catalog_df, project_root=project_root)
    print(f"Extracted feature matrix shape: {embeddings.shape} (L2-normalized)")

    # Save embeddings
    emb_file, meta_file = save_embeddings(
        embeddings=embeddings,
        metadata_df=valid_df,
        output_dir=embeddings_dir,
        filename_prefix="fashion"
    )
    print(f"Saved embeddings to: {emb_file}")

    # 3. Visual Clustering
    print("\n[Step 3] Running Visual Clustering (K-Means)...")
    cluster_engine = VisualClusterEngine(n_clusters=5)
    labels, silhouette = cluster_engine.fit_kmeans(embeddings, find_optimal_k=True, k_range=(3, 6))
    print(f"Optimal clusters: K={cluster_engine.n_clusters}, Silhouette Score: {silhouette:.4f}")

    cluster_summary = cluster_engine.get_cluster_summary(embeddings, valid_df)
    for c in cluster_summary:
        print(f"  • {c['cluster_name']}: {c['item_count']} items | Top styles: {list(c['dominant_styles'].keys())}")

    cluster_engine.save(models_dir / "cluster_model.joblib")

    # Save cluster summary JSON for fast API serving
    with open(forecasts_dir / "clusters_summary.json", "w", encoding="utf-8") as f:
        json.dump(cluster_summary, f, indent=2)

    # 4. Trend Forecasting
    print("\n[Step 4] Running Time-Series Trend Forecaster...")
    forecaster = TrendForecaster(time_series_csv)
    forecast_results = {}
    for style in forecaster.get_available_styles():
        fc = forecaster.forecast_style(style, horizon_weeks=12)
        forecast_results[style] = fc
        print(f"  • {style:<12}: {fc['trajectory']:<20} ({fc['growth_rate_pct']:+.1f}% projected growth)")

    with open(forecasts_dir / "trend_forecasts.json", "w", encoding="utf-8") as f:
        json.dump(forecast_results, f, indent=2)

    # 5. Validate Similarity Engine
    print("\n[Step 5] Validating Similarity Matching Engine...")
    similarity_engine = SimilarityEngine(
        embeddings=embeddings,
        metadata_df=valid_df,
        feature_extractor=extractor
    )

    test_item_id = valid_df.iloc[0]["item_id"]
    test_title = valid_df.iloc[0]["title"]
    sim_matches = similarity_engine.find_similar_by_item_id(test_item_id, top_k=3)
    print(f"Top 3 visual matches for '{test_title}' ({test_item_id}):")
    for m in sim_matches:
        print(f"  -> #{m['rank']} {m['title']} ({m['style']} {m['category']}) - Match: {m['match_percentage']}%")

    print("\n" + "=" * 60)
    print("[SUCCESS] Full Pipeline Completed Successfully!")
    print("=" * 60)

    return {
        "items_count": len(valid_df),
        "embedding_dim": embeddings.shape[1],
        "optimal_k": cluster_engine.n_clusters,
        "silhouette_score": silhouette,
        "forecasted_styles": list(forecast_results.keys())
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    run_full_pipeline(project_root)
