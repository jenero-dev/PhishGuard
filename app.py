"""
app.py — FastAPI backend for PhishGuard
Serves the React dashboard and exposes all API endpoints.
"""

import os
import uuid
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from feature_extractor import FeatureExtractor
from shap_analyzer import ShapAnalyzer  # <-- ADDED

# ── Logging ────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ────
MODEL_PATH         = "models/mlp.pkl"
SCALER_PATH        = "data/processed/scaler.pkl"
FEATURE_NAMES_PATH = "data/processed/feature_names.txt"
TRAIN_X_PATH       = "data/processed/X_train.csv"
INDEX_HTML         = "frontend/index.html"

# ── FastAPI app ────
app = FastAPI(
    title="PhishGuard API",
    description="MSc Information Technology — Phishing Website Detection",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory stores ────
_history: deque = deque(maxlen=500)
_stats = {
    "total_scans": 0,
    "phishing_count": 0,
    "legitimate_count": 0,
    "risk_score_sum": 0.0,
    "daily": {},
}

# ── Helpers ────
def load_feature_names(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature names file not found: {path}")
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def compute_defaults(path: str, names: List[str]) -> Dict[str, float]:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, usecols=names)
            return {n: float(df[n].median()) for n in names}
        except Exception as e:
            logger.warning("Could not compute medians: %s", e)
    return {n: 0.0 for n in names}


def to_python(val):
    if isinstance(val, (np.integer, np.floating, np.bool_)):
        return val.item()
    return val


def get_proba(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[0]
        return float(p[1]) if len(p) >= 2 else float(max(p)), p
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        if isinstance(s, np.ndarray) and s.ndim == 1:
            prob = float(1 / (1 + np.exp(-s[0])))
            return prob, np.array([1 - prob, prob])
        arr = np.asarray(s)[0]
        exps = np.exp(arr - arr.max())
        soft = exps / exps.sum()
        return float(soft[-1]), soft
    pred = int(model.predict(X)[0])
    p = np.array([1.0 - pred, float(pred)])
    return float(p[-1]), p


def record_prediction(entry: dict):
    _history.appendleft(entry)
    _stats["total_scans"] += 1
    if entry["is_phishing"]:
        _stats["phishing_count"] += 1
    else:
        _stats["legitimate_count"] += 1
    _stats["risk_score_sum"] += entry["probability_score"]
    day = entry["timestamp"][:10]
    bucket = _stats["daily"].setdefault(day, {"phishing": 0, "legitimate": 0})
    bucket["phishing" if entry["is_phishing"] else "legitimate"] += 1


# ── Load artefacts at startup ────
logger.info("Loading model artefacts...")
for p in (MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required file missing: {p}. Run training first.")

model            = joblib.load(MODEL_PATH)
scaler           = joblib.load(SCALER_PATH)
trained_features = load_feature_names(FEATURE_NAMES_PATH)
feature_defaults = compute_defaults(TRAIN_X_PATH, trained_features)
extractor        = FeatureExtractor()

# ── Global SHAP via ShapAnalyzer (correctly handles MLP with KernelExplainer) ────
_global_shap: Optional[Dict[str, float]] = None
_shap_analyzer: Optional[ShapAnalyzer] = None

try:
    if os.path.exists(TRAIN_X_PATH):
        df_bg = pd.read_csv(TRAIN_X_PATH, usecols=trained_features).fillna(0).astype(float)
        sample_n = min(200, len(df_bg))
        bg_raw = df_bg.sample(sample_n, random_state=42)
        X_bg = scaler.transform(bg_raw)

        _shap_analyzer = ShapAnalyzer(
            model,
            trained_features,
            background=X_bg,
            max_background=100,
        )

        # Generate global importance; this also writes shap_summary_mlp.png / shap_bar_mlp.png
        global_out_dir = "results/shap"
        os.makedirs(global_out_dir, exist_ok=True)
        _global_shap = _shap_analyzer.global_summary(
            X_bg,
            out_dir=global_out_dir,
            tag="mlp",
            max_samples=200,
        )

        logger.info("Global SHAP importance computed for %d features.", len(_global_shap or {}))
except Exception as e:
    logger.warning("Global SHAP skipped: %s", e)
    import traceback
    logger.debug(traceback.format_exc())


# ── Pydantic models ────
class URLRequest(BaseModel):
    url: str


# ── Core prediction logic (shared) ────
def _predict(url: str) -> dict:
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    if len(url) > 2000:
        raise HTTPException(status_code=400, detail="URL too long")

    raw = extractor.extract_features(url)
    df = pd.DataFrame([raw])
    if "url" in df.columns:
        df = df.drop(columns=["url"])

    for col in trained_features:
        if col not in df.columns:
            df[col] = feature_defaults.get(col, 0.0)
    extra = [c for c in df.columns if c not in trained_features]
    if extra:
        df = df.drop(columns=extra)
    df = df[trained_features].astype(float)

    X_scaled = scaler.transform(df)
    prob, proba = get_proba(model, X_scaled)
    pred = int(model.predict(X_scaled)[0])
    confidence = float(max(proba) * 100)

    return {
        "url": url,
        "is_phishing": bool(pred == 1),
        "confidence": round(confidence, 2),
        "probability_score": round(prob, 4),
        "risk_percent": round(prob * 100, 1),
        "model_used": type(model).__name__,
        "features_extracted": {k: to_python(v) for k, v in raw.items()},
        "_X_scaled": X_scaled,
        "_df_raw": df,
    }


# ── Endpoints ────
@app.get("/")
@app.get("/app")
@app.get("/app/")
async def serve_frontend():
    if os.path.exists(INDEX_HTML):
        return FileResponse(INDEX_HTML, media_type="text/html")
    return HTMLResponse("<h2>index.html not found. Place it next to app.py.</h2>", status_code=404)


@app.post("/predict")
async def predict(request: URLRequest):
    try:
        result = _predict(request.url.strip())
        result.pop("_X_scaled", None)
        result.pop("_df_raw", None)

        ts = datetime.now(timezone.utc).isoformat()
        entry = {**result, "id": str(uuid.uuid4()), "timestamp": ts}
        record_prediction(entry)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/explain")
async def explain(request: URLRequest, top_k: int = Query(default=8, ge=1, le=30)):
    try:
        result = _predict(request.url.strip())
        X_scaled = result.pop("_X_scaled")
        result.pop("_df_raw", None)

        contributions = []
        try:
            if _shap_analyzer is not None and _shap_analyzer.explainer is not None:
                explanation = _shap_analyzer.explain_instance(X_scaled[0], top_k=top_k)
                contributions = [
                    {
                        "feature": c["feature"],
                        "shap_value": round(c["shap_value"], 5),
                    }
                    for c in explanation["contributions"]
                ]
            else:
                logger.warning("Per-URL SHAP skipped: no SHAP analyzer available")
        except Exception as e:
            logger.warning("Per-URL SHAP failed: %s", e)

        ts = datetime.now(timezone.utc).isoformat()
        entry = {**result, "id": str(uuid.uuid4()), "timestamp": ts}
        record_prediction(entry)

        return {
            **result,
            "explanation": {
                "contributions": contributions,
                "note": "Positive SHAP → pushes toward phishing; negative → toward legitimate.",
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Explain error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
async def get_history(limit: int = Query(default=50, ge=1, le=500)):
    items = [
        {k: v for k, v in h.items() if k != "features_extracted"}
        for h in list(_history)[:limit]
    ]
    return {"items": items, "total": len(_history)}


@app.get("/api/stats")
async def get_stats():
    total = _stats["total_scans"]
    avg = (_stats["risk_score_sum"] / total) if total > 0 else 0.0
    daily = [{"day": day, **counts} for day, counts in sorted(_stats["daily"].items())]
    return {
        "total_scans": total,
        "phishing_count": _stats["phishing_count"],
        "legitimate_count": _stats["legitimate_count"],
        "avg_risk_score": round(avg, 4),
        "daily": daily,
    }


@app.get("/api/model-info")
async def model_info():
    return {
        "model": type(model).__name__,
        "num_features": len(trained_features),
        "features": trained_features,
        "global_shap_importance": {
            "feature_importance": _global_shap or {}
        },
    }


# ── Entry point ────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")