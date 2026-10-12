"""
TrendLens — Comprehensive Unit & Integration Tests
Validates:
- FeatureExtractor (Module 2)
- VisualClusterEngine (Module 3)
- TrendForecaster (Module 4)
- SimilarityEngine (Module 5)
- FastAPI Endpoints (Module 6)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.embeddings.extractor import FeatureExtractor
from src.clustering.cluster_engine import VisualClusterEngine
from src.forecasting.trend_forecaster import TrendForecaster
from src.similarity.similarity_engine import SimilarityEngine


@pytest.fixture
def sample_image():
    """Generates a small test RGB image."""
    return Image.new("RGB", (64, 64), color=(180, 50, 70))


@pytest.fixture
def mock_catalog(tmp_path):
    """Generates mock embeddings and metadata for testing."""
    np.random.seed(42)
    n = 10
    d = 128
    embeddings = np.random.randn(n, d)
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    records = []
    for i in range(n):
        records.append({
            "item_id": f"TEST_{i:04d}",
            "title": f"Test Garment {i}",
            "category": "shirt" if i % 2 == 0 else "jacket",
            "style": "Streetwear" if i < 5 else "Y2K",
            "likes": 1000 + i * 100,
            "comments": 50 + i * 5,
            "file_path": f"data/fashion_trends/images/test_{i}.jpg",
            "timestamp": "2024-01-01"
        })
    df = pd.DataFrame(records)
    return embeddings, df


def test_feature_extractor_single_image(sample_image):
    """Verifies that FeatureExtractor generates normalized 1D vector."""
    extractor = FeatureExtractor(backbone_name="resnet18")
    emb = extractor.extract_from_image(sample_image)

    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1
    assert emb.shape[0] == 512
    # Verify unit norm
    norm = np.linalg.norm(emb)
    assert pytest.approx(norm, 1e-4) == 1.0


def test_visual_clustering(mock_catalog):
    """Verifies K-Means fitting, labels, and cluster summary."""
    embeddings, df = mock_catalog
    engine = VisualClusterEngine(n_clusters=3, random_state=42)
    labels, silhouette = engine.fit_kmeans(embeddings, find_optimal_k=False)

    assert len(labels) == len(embeddings)
    assert len(set(labels)) == 3
    assert -1.0 <= silhouette <= 1.0

    summary = engine.get_cluster_summary(embeddings, df)
    assert len(summary) == 3
    assert "exemplar_item" in summary[0]
    assert "dominant_styles" in summary[0]


def test_trend_forecaster(tmp_path):
    """Verifies time-series trajectory projection and classification."""
    # Create synthetic trend csv
    dates = pd.date_range("2024-01-01", periods=30, freq="W")
    trend_rows = []
    for i, d in enumerate(dates):
        trend_rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "style": "Denim",
            "hashtag": "#denim",
            "trend_score": 50.0 + i * 1.5,
            "search_volume": 10000,
            "social_engagement": 5000
        })
    csv_file = tmp_path / "trends.csv"
    pd.DataFrame(trend_rows).to_csv(csv_file, index=False)

    forecaster = TrendForecaster(csv_file)
    assert "Denim" in forecaster.get_available_styles()

    fc = forecaster.forecast_style("Denim", horizon_weeks=8)
    assert fc["style"] == "Denim"
    assert len(fc["forecast_points"]) == 8
    assert "trajectory" in fc
    assert fc["growth_rate_pct"] > 0


def test_similarity_engine(mock_catalog):
    """Verifies cosine similarity Top-K matching."""
    embeddings, df = mock_catalog
    engine = SimilarityEngine(embeddings=embeddings, metadata_df=df)

    # Query using vector of first item
    q_vec = embeddings[0]
    matches = engine.find_similar_by_vector(q_vec, top_k=3)

    assert len(matches) == 3
    assert matches[0]["item_id"] == "TEST_0000"
    assert pytest.approx(matches[0]["similarity_score"], 1e-4) == 1.0
    assert matches[0]["match_percentage"] == 100.0


def test_fastapi_endpoints():
    """Verifies that FastAPI routes respond correctly."""
    from fastapi.testclient import TestClient
    from src.api.main import app

    with TestClient(app) as client:
        health_resp = client.get("/api/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "healthy"

        summary_resp = client.get("/api/trends/summary")
        assert summary_resp.status_code == 200
        assert isinstance(summary_resp.json(), list)

        items_resp = client.get("/api/items?limit=5")
        assert items_resp.status_code == 200
        assert len(items_resp.json()) <= 5
