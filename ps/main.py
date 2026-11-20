"""
FastAPI service for xgboost.json (credit card default model, binary:logistic).

Model source: loaded from Kaggle (manivardhangundoju/credit-default-risk-prediction-model-xgboost/scikitLearn/default/1)
via kagglehub, or from /kaggle/input when run inside a Kaggle notebook, or from a local
xgboost.json if one is present.

Run locally:   pip install fastapi uvicorn xgboost pandas numpy kagglehub
               uvicorn main:app --host 0.0.0.0 --port 8000
Docs (Swagger): http://localhost:8000/docs

Endpoints
  GET  /health         -> service / model status
  POST /predict        -> the 43 model features, exactly as the model was trained
  POST /predict/raw    -> the 23 raw UCI columns; the 20 engineered features are
                          computed server-side by build_features() below
  POST /predict/batch  -> list of the 43-feature rows
"""

import glob
import os
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

# Kaggle Model handle: owner/model/framework/variation/version
# (framework = Scikit-Learn -> "scikitLearn", variation = "default", version = 1)
KAGGLE_HANDLE = os.getenv(
    "KAGGLE_HANDLE",
    "manivardhangundoju/credit-default-risk-prediction-model-xgboost/scikitLearn/default/1",
)
MODEL_FILE = os.getenv("MODEL_FILE", "xgboost.json")
MODEL_PATH = os.getenv("MODEL_PATH", "")  # optional: explicit local path wins
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))  # change to your tuned cut-off

# Must match the order the model was trained with.
FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "utilization_latest", "utilization_mean_6m", "utilization_max_6m",
    "utilization_std_6m", "utilization_trend_6m",
    "avg_bill_6m", "max_bill_6m", "bill_volatility_6m", "bill_trend_slope_6m",
    "total_payment_6m", "avg_payment_6m", "payment_volatility_6m",
    "payment_trend_slope_6m", "payment_active_months_6m",
    "late_month_count_6m", "severe_late_month_count_6m",
    "recent_late_month_count_3m", "max_delay_severity_6m",
    "avg_delay_severity_6m", "consecutive_recent_late_months",
]

state = {}


def _find_file(root: str) -> str | None:
    hits = glob.glob(os.path.join(root, "**", MODEL_FILE), recursive=True)
    return hits[0] if hits else None


def resolve_model_path() -> str:
    """Local path -> Kaggle notebook input -> download from Kaggle with kagglehub."""
    # 1) explicit path or file next to this script
    for p in (MODEL_PATH, MODEL_FILE):
        if p and os.path.isfile(p):
            return p

    # 2) running inside a Kaggle notebook with the model/dataset attached
    if os.path.isdir("/kaggle/input"):
        found = _find_file("/kaggle/input")
        if found:
            return found

    # 3) download from Kaggle (needs `pip install kagglehub` + internet;
    #    outside Kaggle also set KAGGLE_USERNAME / KAGGLE_KEY if the asset is private)
    import kagglehub

    errors = []
    attempts = []
    if KAGGLE_HANDLE.count("/") >= 3:
        attempts.append(kagglehub.model_download)
    else:
        attempts += [kagglehub.dataset_download, kagglehub.model_download]
    for download in attempts:
        try:
            folder = download(KAGGLE_HANDLE)
            found = _find_file(folder)
            if found:
                return found
            errors.append(f"{MODEL_FILE} not found inside {folder}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{download.__name__}: {e}")
    raise RuntimeError(f"Could not get {MODEL_FILE} from '{KAGGLE_HANDLE}': {errors}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = XGBClassifier()
    model.load_model(resolve_model_path())
    state["model"] = model
    yield
    state.clear()


app = FastAPI(title="Credit Default Prediction API", version="1.0.0", lifespan=lifespan)


# ----------------------------- schemas ---------------------------------------
class RawInput(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


class FullInput(RawInput):
    utilization_latest: float
    utilization_mean_6m: float
    utilization_max_6m: float
    utilization_std_6m: float
    utilization_trend_6m: float
    avg_bill_6m: float
    max_bill_6m: float
    bill_volatility_6m: float
    bill_trend_slope_6m: float
    total_payment_6m: float
    avg_payment_6m: float
    payment_volatility_6m: float
    payment_trend_slope_6m: float
    payment_active_months_6m: int
    late_month_count_6m: int
    severe_late_month_count_6m: int
    recent_late_month_count_3m: int
    max_delay_severity_6m: int
    avg_delay_severity_6m: float
    consecutive_recent_late_months: int


class Prediction(BaseModel):
    default_probability: float = Field(..., description="P(default next month)")
    prediction: int = Field(..., description="1 = default, 0 = no default")
    threshold: float


class BatchPrediction(BaseModel):
    predictions: List[Prediction]


# ----------------------------- feature engineering ---------------------------
def _slope(y: np.ndarray) -> float:
    """Linear-trend slope over time (oldest -> newest)."""
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def build_features(r: RawInput) -> dict:
    """
    Derive the 20 engineered features from the 23 raw columns.

    NOTE: these formulas are reconstructed from the feature names. If your
    training notebook defines them differently, edit them here so they match
    EXACTLY -- otherwise predictions from /predict/raw will be off.
    Month order in the UCI data: *1 = most recent (Sep), *6 = oldest (Apr).
    """
    limit = max(r.LIMIT_BAL, 1.0)
    bills = np.array([r.BILL_AMT6, r.BILL_AMT5, r.BILL_AMT4,
                      r.BILL_AMT3, r.BILL_AMT2, r.BILL_AMT1], dtype=float)  # old -> new
    pays = np.array([r.PAY_AMT6, r.PAY_AMT5, r.PAY_AMT4,
                     r.PAY_AMT3, r.PAY_AMT2, r.PAY_AMT1], dtype=float)      # old -> new
    status = np.array([r.PAY_0, r.PAY_2, r.PAY_3, r.PAY_4, r.PAY_5, r.PAY_6])  # new -> old
    util = bills / limit                                                        # old -> new
    delay = np.clip(status, 0, None)                                            # -1/0 => not late

    late = status > 0
    consecutive = 0
    for is_late in late:  # starts from the most recent month
        if not is_late:
            break
        consecutive += 1

    return {
        "utilization_latest": float(util[-1]),
        "utilization_mean_6m": float(util.mean()),
        "utilization_max_6m": float(util.max()),
        "utilization_std_6m": float(util.std()),
        "utilization_trend_6m": _slope(util),
        "avg_bill_6m": float(bills.mean()),
        "max_bill_6m": float(bills.max()),
        "bill_volatility_6m": float(bills.std()),
        "bill_trend_slope_6m": _slope(bills),
        "total_payment_6m": float(pays.sum()),
        "avg_payment_6m": float(pays.mean()),
        "payment_volatility_6m": float(pays.std()),
        "payment_trend_slope_6m": _slope(pays),
        "payment_active_months_6m": int((pays > 0).sum()),
        "late_month_count_6m": int(late.sum()),
        "severe_late_month_count_6m": int((status >= 2).sum()),
        "recent_late_month_count_3m": int(late[:3].sum()),
        "max_delay_severity_6m": int(delay.max()),
        "avg_delay_severity_6m": float(delay.mean()),
        "consecutive_recent_late_months": consecutive,
    }


# ----------------------------- helpers ---------------------------------------
def _predict(rows: List[dict]) -> List[Prediction]:
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = pd.DataFrame(rows)[FEATURES]
    probs = model.predict_proba(df)[:, 1]
    return [
        Prediction(
            default_probability=round(float(p), 6),
            prediction=int(p >= THRESHOLD),
            threshold=THRESHOLD,
        )
        for p in probs
    ]


# ----------------------------- routes ----------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state, "n_features": len(FEATURES)}


@app.post("/predict", response_model=Prediction)
def predict(payload: FullInput):
    return _predict([payload.model_dump()])[0]


@app.post("/predict/raw", response_model=Prediction)
def predict_raw(payload: RawInput):
    row = {**payload.model_dump(), **build_features(payload)}
    return _predict([row])[0]


@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch(payload: List[FullInput]):
    if not payload:
        raise HTTPException(status_code=422, detail="Empty list")
    return BatchPrediction(predictions=_predict([p.model_dump() for p in payload]))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
