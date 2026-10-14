"""
TrendLens - Automotive Pipeline Runner
Executes the automotive POC pipeline using the existing TrendLens
ML components (FeatureExtractor, VisualClusterEngine, SimilarityEngine).

Steps:
  1. Load and validate the automotive smoke-test catalog (AutomotiveAdapter)
  2. Extract deep visual embeddings (FeatureExtractor / ResNet-18)
  3. Fit visual clusters (VisualClusterEngine / K-Means)
  4. Validate similarity search (SimilarityEngine)
  5. Save all artifacts to data/automotive/ and models/

NOTE: Trend forecasting is INTENTIONALLY DISABLED.
  model_year labels encode production year, not sales/popularity.
  The API returns a static placeholder instead of a forecast chart.
"""

import json
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.embeddings.extractor import FeatureExtractor, save_embeddings
from src.clustering.cluster_engine import VisualClusterEngine
from src.similarity.similarity_engine import SimilarityEngine
from src.preprocessing.automotive_adapter import AutomotiveAdapter


def run_automotive_pipeline(project_root: Path) -> dict:
    print("=" * 60)
    print(">> Running TrendLens Automotive POC Pipeline")
    print("=" * 60)

    # Directories
    embeddings_dir = project_root / "data" / "automotive" / "embeddings"
    models_dir = project_root / "models"
    forecasts_dir = project_root / "data" / "automotive" / "forecasts"

    embeddings_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    forecasts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load automotive catalog
    print("\n[Step 1] Loading automotive catalog (AutomotiveAdapter)...")
    adapter = AutomotiveAdapter(project_root)
    catalog_df = adapter.build_catalog(skip_missing=True)
    adapter.save_catalog(catalog_df)

    dist = adapter.get_body_type_distribution(catalog_df)
    print(f"  Body-type distribution: {dist}")

    if len(catalog_df) < 4:
        raise RuntimeError(
            f"Only {len(catalog_df)} images found. Need >= 4 for clustering.\n"
            "Populate data/automotive/smoke_test/images/ with Stanford Cars images."
        )

    # 2. Extract visual embeddings
    print(f"\n[Step 2] Extracting visual embeddings for {len(catalog_df)} vehicles...")
    extractor = FeatureExtractor(backbone_name="resnet18")
    embeddings, valid_df = extractor.extract_from_catalog(
        catalog_df, project_root=project_root
    )
    print(f"  Embedding matrix: {embeddings.shape} (L2-normalized ResNet-18)")

    emb_file, meta_file = save_embeddings(
        embeddings=embeddings,
        metadata_df=valid_df,
        output_dir=embeddings_dir,
        filename_prefix="automotive"
    )
    print(f"  Saved embeddings -> {emb_file}")

    # 3. Visual clustering
    n_samples = len(embeddings)
    max_k = min(6, n_samples - 1)
    min_k = min(3, max_k)
    print(f"\n[Step 3] Running K-Means clustering (k_range={min_k}-{max_k})...")
    cluster_engine = VisualClusterEngine(n_clusters=min_k)
    labels, silhouette = cluster_engine.fit_kmeans(
        embeddings, find_optimal_k=(n_samples >= 8), k_range=(min_k, max_k)
    )
    print(
        f"  Optimal K={cluster_engine.n_clusters}, "
        f"Silhouette={silhouette:.4f}"
    )

    cluster_summary = cluster_engine.get_cluster_summary(embeddings, valid_df)
    for c in cluster_summary:
        body_types = list(c["dominant_styles"].keys())
        print(
            f"  * {c['cluster_name']}: {c['item_count']} vehicles | "
            f"Body types: {body_types}"
        )

    cluster_engine.save(models_dir / "automotive_cluster_model.joblib")

    with open(forecasts_dir / "automotive_clusters_summary.json", "w", encoding="utf-8") as f:
        json.dump(cluster_summary, f, indent=2)
    print(f"  Saved cluster summary -> {forecasts_dir / 'automotive_clusters_summary.json'}")

    # 4. Forecasting placeholder
    print("\n[Step 4] Trend Forecasting: DISABLED (no temporal sales data)")
    forecasting_status = {
        "status": "disabled",
        "reason": (
            "Automotive forecasting requires real sales, search-volume, or "
            "registration data with temporal coverage. The Stanford Cars "
            "model_year labels encode production year only and must NOT be "
            "treated as a popularity or trend signal."
        ),
        "data_needed": [
            "Monthly/weekly sales figures by model (e.g. BLS / OEM reports)",
            "Google Trends search volumes by model name",
            "Vehicle registration statistics (e.g. DVLA / DMV open data)",
        ]
    }
    with open(forecasts_dir / "automotive_forecast_status.json", "w", encoding="utf-8") as f:
        json.dump(forecasting_status, f, indent=2)

    # 5. Validate similarity engine
    print("\n[Step 5] Validating Similarity Engine...")
    similarity_engine = SimilarityEngine(
        embeddings=embeddings,
        metadata_df=valid_df,
        feature_extractor=extractor
    )

    test_item = valid_df.iloc[0]
    sim_matches = similarity_engine.find_similar_by_item_id(test_item["item_id"], top_k=3)
    print(f"  Top-3 matches for '{test_item['title']}' ({test_item['item_id']}):")
    for m in sim_matches:
        print(f"    -> #{m['rank']} {m['title']} ({m['category']}) - {m['match_percentage']}% match")

    print("\n" + "=" * 60)
    print("[SUCCESS] Automotive POC Pipeline Completed!")
    print("=" * 60)

    return {
        "items_count": len(valid_df),
        "embedding_dim": embeddings.shape[1],
        "optimal_k": cluster_engine.n_clusters,
        "silhouette_score": round(silhouette, 4),
        "body_type_distribution": dist,
        "forecasting_status": "disabled"
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    result = run_automotive_pipeline(project_root)
    print("\nPipeline Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
