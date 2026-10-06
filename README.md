# TrendLens — A Multi-Genre AI-Powered Visual Trend Analysis & Forecasting Platform

**BSCS Final Year Project (FYP)**

---

## 📌 Project Overview
**TrendLens** is an AI-powered visual trend analysis and forecasting platform. The platform analyzes product images across different domains (starting with **Fashion** and later demonstrating a proof-of-concept in **Automobiles**), extracts deep visual features, groups visually similar products into clusters, tracks how clusters evolve over time, forecasts trend trajectory, and allows users to match new images against active trends.

---

## 🚀 Core Pipeline
```text
Product Images
      │
      ▼
1. Preprocessing (Resize, Normalize, Data Validation)
      │
      ▼
2. Visual Feature Extraction (CNN/ResNet or CLIP Embeddings)
      │
      ▼
3. Visual Clustering (K-Means / DBSCAN)
      │
      ▼
4. Trend Identification & Temporal Aggregation
      │
      ▼
5. Time-Series Forecasting (Prophet)
      │
      ▼
6. Similarity Engine (Cosine Similarity matching)
      │
      ▼
7. Application Layer (FastAPI Backend + PostgreSQL + Web Dashboard + WordPress Reports)
```

---

## 🛠️ Technology Stack
- **Languages:** Python (Core Engine & Backend), PHP / JavaScript (Web Presentation)
- **Deep Learning / Vision:** PyTorch, torchvision (ResNet / CLIP)
- **Machine Learning & Analytics:** scikit-learn (K-Means, DBSCAN, Cosine Similarity), Prophet (Time-series)
- **Backend API:** FastAPI, Uvicorn, Pydantic
- **Database:** PostgreSQL (with SQLAlchemy)
- **CMS / Reporting:** WordPress, PHP
- **Testing & Code Quality:** pytest

---

## 📂 Project Structure
```text
trendlens/
├── data/                    # Datasets (git-ignored, strictly structured)
│   ├── raw/                 # Original image downloads (DeepFashion2 / Stanford Cars)
│   ├── processed/           # Resized and cleaned images
│   ├── embeddings/          # Extracted visual feature vectors (.npy / .parquet)
│   ├── forecasts/           # Trend forecasting models & outputs
│   └── samples/             # Small test images for unit testing
├── docs/                    # Architecture diagrams, design docs, FYP presentation notes
├── models/                  # Pretrained model weights & cluster models (git-ignored)
├── src/
│   ├── preprocessing/       # Module 1: Image loading, transformation, metadata parsing
│   ├── embeddings/          # Module 2: CNN / CLIP feature extraction
│   ├── clustering/          # Module 3: K-Means & DBSCAN cluster pipelines
│   ├── forecasting/         # Module 4: Time-series trend volume forecasting (Prophet)
│   ├── similarity/          # Module 5: Image search and cosine similarity matching
│   └── api/                 # Module 6: FastAPI endpoints and schemas
├── tests/                   # Automated unit & integration tests per module
├── .gitignore               # Strict ignore rules for large data and weights
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 📋 Module Development Roadmap

| Module | Name | Focus / Deliverable | Status |
| :---: | :--- | :--- | :---: |
| **0** | **Environment & Foundation** | Git setup, virtual env, folder layout, dependency verification | 🟡 *In Progress* |
| **1** | **Dataset & Preprocessing** | DeepFashion2 sample ingestion, validation, image transformations | ⚪ *Pending* |
| **2** | **Visual Feature Extraction** | CNN (ResNet) / CLIP embedding pipeline & vector export | ⚪ *Pending* |
| **3** | **Visual Clustering** | K-Means & DBSCAN clustering, cluster evaluation metrics (Silhouette) | ⚪ *Pending* |
| **4** | **Trend Identification & Forecasting** | Temporal aggregation, cluster popularity over time, Prophet forecasting | ⚪ *Pending* |
| **5** | **Similarity Matching Engine** | Query image vector extraction & Top-K cosine similarity matching | ⚪ *Pending* |
| **6** | **FastAPI Backend & PostgreSQL** | REST API endpoints, image upload, DB models for trends | ⚪ *Pending* |
| **7** | **Dashboard & WordPress Integration** | Frontend visualizations and WordPress report generation | ⚪ *Pending* |
| **8** | **Automobile POC (Secondary)** | Applying the validated pipeline to Stanford Cars / CompCars | ⚪ *Pending* |

---

## ⚙️ Initial Setup Guide

### 1. Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Project Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run Automated Tests
```powershell
pytest
```
