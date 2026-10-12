"""
TrendLens — Module 6: FastAPI Application & REST API
Exposes endpoints for:
- Historical & projected trend analytics
- Visual style clusters & exemplars
- Fashion catalog explorer
- Real-time image upload & visual similarity search
- Serves the modern web frontend dashboard
"""

import io
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.clustering.cluster_engine import VisualClusterEngine
from src.embeddings.extractor import FeatureExtractor, load_embeddings
from src.forecasting.trend_forecaster import TrendForecaster
from src.similarity.similarity_engine import SimilarityEngine

from contextlib import asynccontextmanager

# Global engines
extractor: Optional[FeatureExtractor] = None
forecaster: Optional[TrendForecaster] = None
similarity_engine: Optional[SimilarityEngine] = None
cluster_summary_cache: Optional[List[Dict]] = None
catalog_df: Optional[pd.DataFrame] = None


def load_app_resources():
    global extractor, forecaster, similarity_engine, cluster_summary_cache, catalog_df

    trends_csv = PROJECT_ROOT / "data" / "fashion_trends" / "insta_fashion_trends.csv"
    meta_csv = PROJECT_ROOT / "data" / "fashion_trends" / "fashion_posts_metadata.csv"
    embeddings_dir = PROJECT_ROOT / "data" / "embeddings"
    forecasts_dir = PROJECT_ROOT / "data" / "forecasts"

    # Initialize forecaster
    if trends_csv.is_file():
        forecaster = TrendForecaster(trends_csv)

    # Initialize catalog
    if meta_csv.is_file():
        catalog_df = pd.read_csv(meta_csv)

    # Initialize cached cluster summary
    clusters_json = forecasts_dir / "clusters_summary.json"
    if clusters_json.is_file():
        with open(clusters_json, "r", encoding="utf-8") as f:
            cluster_summary_cache = json.load(f)

    # Initialize feature extractor & similarity engine
    emb_file = embeddings_dir / "fashion_embeddings.npy"
    if emb_file.is_file() and catalog_df is not None:
        try:
            embeddings, meta = load_embeddings(embeddings_dir, filename_prefix="fashion")
            extractor = FeatureExtractor(backbone_name="resnet18")
            similarity_engine = SimilarityEngine(
                embeddings=embeddings,
                metadata_df=meta,
                feature_extractor=extractor
            )
        except Exception as e:
            print(f"Warning during similarity engine init: {e}")


def ensure_resources_loaded():
    if forecaster is None or catalog_df is None:
        load_app_resources()


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    load_app_resources()
    yield


app = FastAPI(
    title="TrendLens AI Visual Trend & Forecasting API",
    description="Multimodal Fashion Trend Forecasting, Clustering, and Visual Search Engine",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local web development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Static files mount
images_dir = PROJECT_ROOT / "data" / "fashion_trends" / "images"
if images_dir.is_dir():
    app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")

web_dir = PROJECT_ROOT / "web"
if web_dir.is_dir():
    app.mount("/web", StaticFiles(directory=str(web_dir)), name="web")


@app.get("/")
def serve_home():
    """Serves the main website dashboard."""
    index_file = PROJECT_ROOT / "web" / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return {"message": "TrendLens API is running. Web directory not yet initialized."}


@app.get("/api/health")
def health_check():
    ensure_resources_loaded()
    return {
        "status": "healthy",
        "forecaster_ready": forecaster is not None,
        "similarity_ready": similarity_engine is not None,
        "catalog_size": len(catalog_df) if catalog_df is not None else 0
    }


@app.get("/api/trends/summary")
def get_trends_summary():
    """Returns top trend momentum across all monitored fashion styles."""
    ensure_resources_loaded()
    if forecaster is None:
        raise HTTPException(status_code=503, detail="Trend forecasting engine initializing.")
    return forecaster.get_all_trends_summary()


@app.get("/api/trends/forecast")
def get_style_forecast(
    style: str = Query("Streetwear", description="Fashion style name"),
    horizon: int = Query(12, ge=4, le=26, description="Forecast horizon in weeks")
):
    """Returns time-series historical points and future projections with confidence bands."""
    if forecaster is None:
        raise HTTPException(status_code=503, detail="Forecasting engine not ready.")
    try:
        return forecaster.forecast_style(style=style, horizon_weeks=horizon)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/clusters")
def get_clusters():
    """Returns visual style clusters and representative exemplar items."""
    if cluster_summary_cache is not None:
        return cluster_summary_cache

    forecasts_dir = PROJECT_ROOT / "data" / "forecasts"
    clusters_json = forecasts_dir / "clusters_summary.json"
    if clusters_json.is_file():
        with open(clusters_json, "r", encoding="utf-8") as f:
            return json.load(f)

    raise HTTPException(status_code=503, detail="Clustering pipeline output not found. Run pipeline_runner.py first.")


@app.get("/api/items")
def get_catalog_items(
    style: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 50
):
    """Returns list of fashion catalog items with optional filtering."""
    if catalog_df is None:
        raise HTTPException(status_code=503, detail="Catalog metadata not loaded.")

    df = catalog_df.copy()
    if style:
        df = df[df["style"].str.lower() == style.lower()]
    if category:
        df = df[df["category"].str.lower() == category.lower()]

    return df.head(limit).to_dict(orient="records")


@app.post("/api/similarity/match")
async def match_image_similarity(
    file: Optional[UploadFile] = File(None),
    item_id: Optional[str] = Query(None),
    top_k: int = Query(6, ge=1, le=12)
):
    """
    Finds visually similar fashion items via deep cosine similarity.
    Accepts either an uploaded image or an existing item_id.
    """
    if similarity_engine is None:
        raise HTTPException(status_code=503, detail="Visual similarity engine not ready. Run pipeline_runner.py first.")

    if file is not None:
        try:
            contents = await file.read()
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
            matches = similarity_engine.find_similar_by_image(pil_img, top_k=top_k)
            return {
                "query_source": "image_upload",
                "filename": file.filename,
                "matches": matches
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    elif item_id:
        try:
            matches = similarity_engine.find_similar_by_item_id(item_id, top_k=top_k)
            return {
                "query_source": "catalog_item",
                "item_id": item_id,
                "matches": matches
            }
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    else:
        raise HTTPException(status_code=400, detail="Must provide either an uploaded image file or item_id parameter.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
