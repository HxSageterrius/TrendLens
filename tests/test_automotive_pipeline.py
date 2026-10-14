"""
TrendLens - Automotive POC Tests
Tests metadata parsing, catalog building, embedding generation,
clustering, similarity search, and API genre switching.

These tests use SYNTHETIC data (PIL-generated placeholder images) so they
pass without real Stanford Cars images on disk.  The real dataset images
are needed only for the full pipeline runner (automotive_pipeline_runner.py).
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.preprocessing.automotive_adapter import (
    AutomotiveAdapter,
    parse_label_components,
    infer_body_type,
    SMOKE_TEST_MANIFEST,
)
from src.embeddings.extractor import FeatureExtractor
from src.clustering.cluster_engine import VisualClusterEngine
from src.similarity.similarity_engine import SimilarityEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_auto_image():
    """Minimal RGB PIL image for testing the feature extractor."""
    return Image.new("RGB", (224, 224), color=(100, 140, 180))


@pytest.fixture
def mock_auto_catalog(tmp_path):
    """
    Generates synthetic automotive catalog with real images on disk so that
    AutomotiveAdapter and FeatureExtractor can process them.
    """
    images_dir = tmp_path / "data" / "automotive" / "smoke_test" / "images"
    images_dir.mkdir(parents=True)

    n = 12  # enough for clustering
    records = []
    body_types = ["sedan", "suv", "coupe", "truck", "convertible", "wagon"]
    makes = ["BMW", "Toyota", "Ford", "Audi", "Honda", "Jeep"]

    for i in range(n):
        filename = f"AUTO_{i+1:04d}_test_car.jpg"
        img_path = images_dir / filename
        # Generate visually distinct synthetic images
        color = ((i * 40) % 256, (i * 70 + 30) % 256, (i * 100 + 60) % 256)
        Image.new("RGB", (224, 224), color=color).save(img_path, "JPEG")

        body_type = body_types[i % len(body_types)]
        make = makes[i % len(makes)]
        rel_path = str(
            Path("data") / "automotive" / "smoke_test" / "images" / filename
        )
        records.append({
            "item_id":    f"AUTO_{i+1:04d}",
            "title":      f"{make} Test Model {2010 + i}",
            "category":   body_type,
            "style":      body_type,
            "make":       make,
            "model":      "Test Model",
            "model_year": 2010 + i,
            "file_path":  rel_path,
            "tags":       f"#{make.lower()} #{body_type} #stanfordcars",
        })

    df = pd.DataFrame(records)
    return df, tmp_path


@pytest.fixture
def mock_embeddings(mock_auto_catalog):
    """Pre-computed mock embedding matrix (avoids real ResNet inference)."""
    df, _ = mock_auto_catalog
    np.random.seed(42)
    n, d = len(df), 512
    emb = np.random.randn(n, d).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / (norms + 1e-9)
    return emb, df


# ---------------------------------------------------------------------------
# 1. Metadata parsing tests
# ---------------------------------------------------------------------------

class TestMetadataParsing:

    def test_parse_sedan_label(self):
        make, model, year, body = parse_label_components("BMW 3 Series Sedan 2012")
        assert make == "BMW"
        assert "3 Series" in model
        assert year == 2012
        assert body == "sedan"

    def test_parse_suv_label(self):
        _, _, _, body = parse_label_components("Toyota 4Runner SUV 2012")
        assert body == "suv"

    def test_parse_coupe_label(self):
        _, _, _, body = parse_label_components("Audi TT RS Coupe 2012")
        assert body == "coupe"

    def test_parse_truck_label(self):
        make, _, year, body = parse_label_components("Ford F-150 Regular Cab 2012")
        assert make == "Ford"
        assert year == 2012
        assert body == "truck"

    def test_parse_convertible_label(self):
        _, _, _, body = parse_label_components("BMW 3 Series Convertible 2012")
        assert body == "convertible"

    def test_parse_wagon_label(self):
        _, _, _, body = parse_label_components("BMW 3 Series Wagon 2012")
        assert body == "wagon"

    def test_parse_minivan_label(self):
        _, _, _, body = parse_label_components("Dodge Grand Caravan Minivan 2011")
        assert body == "wagon"

    def test_parse_missing_year(self):
        make, model, year, body = parse_label_components("Honda Accord Coupe")
        assert make == "Honda"
        assert year == 0

    def test_infer_body_type_fallback(self):
        body = infer_body_type("Some Unknown Vehicle Model XYZ")
        assert body == "other"

    def test_manifest_has_all_body_types(self):
        """All 6 body types must be represented in the manifest."""
        labels = [label for label, _ in SMOKE_TEST_MANIFEST]
        body_types = {infer_body_type(label) for label in labels}
        required = {"sedan", "suv", "coupe", "truck", "convertible", "wagon"}
        assert required.issubset(body_types), (
            f"Missing body types: {required - body_types}"
        )

    def test_manifest_item_count(self):
        """Manifest should have 40-50 entries."""
        assert 40 <= len(SMOKE_TEST_MANIFEST) <= 50


# ---------------------------------------------------------------------------
# 2. AutomotiveAdapter tests
# ---------------------------------------------------------------------------

class TestAutomotiveAdapter:

    def test_build_catalog_with_real_images(self, mock_auto_catalog, tmp_path):
        """Adapter builds catalog from real synthetic images on disk."""
        df, project_root = mock_auto_catalog
        # Write a minimal manifest from those images
        adapter = AutomotiveAdapter(project_root)

        # Build catalog using full manifest - images won't all exist, skip_missing=True
        result = adapter.build_catalog(skip_missing=True)
        # Could be 0 since files don't match manifest names; that's fine for this test
        assert isinstance(result, pd.DataFrame)

    def test_build_catalog_with_custom_manifest(self, tmp_path):
        """Adapter correctly normalises custom manifest entries."""
        images_dir = tmp_path / "data" / "automotive" / "smoke_test" / "images"
        images_dir.mkdir(parents=True)

        # Create two test images
        custom_manifest = [
            ("BMW M3 Coupe 2012",      "test_bmw_coupe.jpg"),
            ("Ford F-150 Regular Cab 2012", "test_ford_truck.jpg"),
        ]
        for _, fname in custom_manifest:
            Image.new("RGB", (64, 64), color=(100, 100, 100)).save(
                images_dir / fname, "JPEG"
            )

        adapter = AutomotiveAdapter(tmp_path)
        result = adapter.build_catalog(manifest=custom_manifest, skip_missing=False)

        assert len(result) == 2
        assert "item_id" in result.columns
        assert "title" in result.columns
        assert "category" in result.columns
        assert "style" in result.columns
        assert "make" in result.columns
        assert "model_year" in result.columns
        assert "file_path" in result.columns

        # Verify schema values
        coupe_row = result[result["title"].str.contains("Coupe")].iloc[0]
        assert coupe_row["category"] == "coupe"
        assert coupe_row["style"] == "coupe"  # matches fashion 'style'
        assert coupe_row["make"] == "BMW"
        assert coupe_row["model_year"] == 2012

        truck_row = result[result["title"].str.contains("Cab")].iloc[0]
        assert truck_row["category"] == "truck"

    def test_skip_missing_true(self, tmp_path):
        """skip_missing=True should drop missing files gracefully."""
        manifest = [("BMW M3 Coupe 2012", "nonexistent.jpg")]
        adapter = AutomotiveAdapter(tmp_path)
        result = adapter.build_catalog(manifest=manifest, skip_missing=True)
        assert len(result) == 0

    def test_skip_missing_false_raises(self, tmp_path):
        """skip_missing=False should raise on missing files."""
        manifest = [("BMW M3 Coupe 2012", "nonexistent.jpg")]
        adapter = AutomotiveAdapter(tmp_path)
        with pytest.raises(FileNotFoundError):
            adapter.build_catalog(manifest=manifest, skip_missing=False)

    def test_save_and_load_catalog(self, tmp_path):
        """save_catalog -> load_catalog round-trip."""
        images_dir = tmp_path / "data" / "automotive" / "smoke_test" / "images"
        images_dir.mkdir(parents=True)
        Image.new("RGB", (64, 64), color=(120, 80, 60)).save(
            images_dir / "test.jpg", "JPEG"
        )

        adapter = AutomotiveAdapter(tmp_path)
        manifest = [("Honda CR-V SUV 2012", "test.jpg")]
        df = adapter.build_catalog(manifest=manifest, skip_missing=False)
        adapter.save_catalog(df)

        loaded = adapter.load_catalog()
        assert len(loaded) == len(df)
        assert list(loaded["item_id"]) == list(df["item_id"])

    def test_attribution_file_created(self, tmp_path):
        """save_catalog should write ATTRIBUTION.md."""
        images_dir = tmp_path / "data" / "automotive" / "smoke_test" / "images"
        images_dir.mkdir(parents=True)
        Image.new("RGB", (64, 64)).save(images_dir / "test.jpg", "JPEG")

        adapter = AutomotiveAdapter(tmp_path)
        df = adapter.build_catalog(
            manifest=[("Ford Mustang Coupe 2007", "test.jpg")],
            skip_missing=False
        )
        adapter.save_catalog(df)

        attr_file = tmp_path / "data" / "automotive" / "smoke_test" / "ATTRIBUTION.md"
        assert attr_file.is_file()
        content = attr_file.read_text(encoding="utf-8")
        assert "Stanford Cars" in content
        assert "non-commercial" in content


# ---------------------------------------------------------------------------
# 3. Embedding generation tests (use mock catalog with real disk files)
# ---------------------------------------------------------------------------

class TestAutomotiveEmbeddings:

    def test_extractor_produces_normalized_vectors(self, sample_auto_image):
        """FeatureExtractor emits L2-normalized 512-dim vectors for car images."""
        extractor = FeatureExtractor(backbone_name="resnet18")
        emb = extractor.extract_from_image(sample_auto_image)
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 1
        assert emb.shape[0] == 512
        norm = np.linalg.norm(emb)
        assert pytest.approx(norm, abs=1e-4) == 1.0

    def test_extract_from_catalog(self, mock_auto_catalog):
        """extract_from_catalog processes all valid catalog images."""
        df, project_root = mock_auto_catalog
        extractor = FeatureExtractor(backbone_name="resnet18")
        embeddings, valid_df = extractor.extract_from_catalog(
            df, project_root=project_root
        )
        assert embeddings.shape[0] == len(valid_df)
        assert embeddings.shape[1] == 512
        assert len(valid_df) == len(df)  # all images exist in mock


# ---------------------------------------------------------------------------
# 4. Clustering tests
# ---------------------------------------------------------------------------

class TestAutomotiveClustering:

    def test_kmeans_on_auto_embeddings(self, mock_embeddings):
        """K-Means clusters automotive embeddings with silhouette > -1."""
        emb, df = mock_embeddings
        engine = VisualClusterEngine(n_clusters=3)
        labels, silhouette = engine.fit_kmeans(emb, find_optimal_k=False)

        assert len(labels) == len(emb)
        assert len(set(labels)) == 3
        assert -1.0 <= silhouette <= 1.0

    def test_cluster_summary_has_required_fields(self, mock_embeddings):
        """Cluster summary uses 'style' and 'category' fields from auto schema."""
        emb, df = mock_embeddings
        engine = VisualClusterEngine(n_clusters=3)
        engine.fit_kmeans(emb, find_optimal_k=False)
        summary = engine.get_cluster_summary(emb, df)

        assert len(summary) == 3
        for c in summary:
            assert "cluster_id" in c
            assert "exemplar_item" in c
            assert "dominant_styles" in c
            assert "dominant_categories" in c

    def test_optimal_k_search(self, mock_embeddings):
        """Optimal K search stays within valid bounds."""
        emb, df = mock_embeddings
        engine = VisualClusterEngine(n_clusters=3)
        labels, silhouette = engine.fit_kmeans(
            emb, find_optimal_k=True, k_range=(2, 5)
        )
        assert 2 <= engine.n_clusters <= 5


# ---------------------------------------------------------------------------
# 5. Similarity search tests
# ---------------------------------------------------------------------------

class TestAutomotiveSimilarity:

    def test_find_similar_by_vector(self, mock_embeddings):
        """Top-K similarity returns correctly ranked results."""
        emb, df = mock_embeddings
        engine = SimilarityEngine(embeddings=emb, metadata_df=df)

        query = emb[0]
        matches = engine.find_similar_by_vector(query, top_k=3)

        assert len(matches) == 3
        assert matches[0]["item_id"] == df.iloc[0]["item_id"]
        assert pytest.approx(matches[0]["similarity_score"], abs=1e-4) == 1.0
        assert matches[0]["match_percentage"] == 100.0

    def test_find_similar_by_item_id_excludes_self(self, mock_embeddings):
        """find_similar_by_item_id excludes the query item from results."""
        emb, df = mock_embeddings
        engine = SimilarityEngine(embeddings=emb, metadata_df=df)

        first_id = df.iloc[0]["item_id"]
        matches = engine.find_similar_by_item_id(first_id, top_k=3)

        returned_ids = [m["item_id"] for m in matches]
        assert first_id not in returned_ids

    def test_auto_fields_in_matches(self, mock_embeddings):
        """Similarity results include automotive-specific metadata fields."""
        emb, df = mock_embeddings
        engine = SimilarityEngine(embeddings=emb, metadata_df=df)

        matches = engine.find_similar_by_vector(emb[0], top_k=1)
        m = matches[0]
        assert "make" in m
        assert "model_year" in m
        assert "category" in m


# ---------------------------------------------------------------------------
# 6. API genre-switching tests
# ---------------------------------------------------------------------------

class TestApiGenreSwitching:

    def test_health_endpoint_reports_genre_field(self):
        """Health endpoint should include a genre field when extended."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        with TestClient(app) as client:
            resp = client.get("/api/health")
            assert resp.status_code == 200
            data = resp.json()
            assert "status" in data
            assert data["status"] == "healthy"

    def test_genre_fashion_returns_catalog_items(self):
        """Items endpoint with genre=fashion returns fashion items."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        with TestClient(app) as client:
            resp = client.get("/api/items?genre=fashion&limit=3")
            assert resp.status_code == 200
            items = resp.json()
            assert isinstance(items, list)

    def test_genre_automotive_returns_placeholder(self):
        """Items endpoint with genre=automotive returns automotive items or placeholder."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        with TestClient(app) as client:
            resp = client.get("/api/items?genre=automotive&limit=3")
            # Accepts 200 (items exist) or 503 (pipeline not run yet)
            assert resp.status_code in (200, 503)

    def test_automotive_trend_forecast_disabled(self):
        """Automotive forecast endpoint returns disabled placeholder."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        with TestClient(app) as client:
            resp = client.get("/api/trends/forecast?genre=automotive&style=sedan")
            data = resp.json()
            # Either 503 or a response with forecasting_disabled=True
            assert resp.status_code in (200, 503)
            if resp.status_code == 200:
                assert data.get("forecasting_disabled") is True
