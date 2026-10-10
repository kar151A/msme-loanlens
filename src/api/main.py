
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "loan_model.pkl"


class LoanApplication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    business_vintage_months: int = Field(ge=0, le=600)
    annual_revenue_lakhs: float = Field(gt=0)
    existing_debt_lakhs: float = Field(ge=0)
    cibil_score: int = Field(ge=300, le=900)
    sector: str = Field(
        pattern="^(manufacturing|services|trading|agriculture)$"
    )
    loan_amount_lakhs: float = Field(gt=0)


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Trained model missing. Run src/ml/train_model.py first."
        )
    return joblib.load(MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield


app = FastAPI(
    title="MSME LoanLens API",
    description="Explainable MSME loan prediction prototype",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "healthy", "service": "MSME LoanLens"}


@app.post("/predict")
def predict(application: LoanApplication):
    try:
        model = get_model()

        features = application.model_dump()
        revenue = features["annual_revenue_lakhs"]

        features["debt_to_revenue_ratio"] = round(
            features["existing_debt_lakhs"] / revenue, 2
        )
        features["loan_to_revenue_ratio"] = round(
            features["loan_amount_lakhs"] / revenue, 2
        )

        data = pd.DataFrame([features])

        decision = str(model.predict(data)[0])
        probabilities = model.predict_proba(data)[0]
        classes = list(model.classes_)
        approval_index = classes.index("approved")

        return {
            "decision": decision,
            "approval_probability": round(
                float(probabilities[approval_index]), 4
            ),
            "model_type": "RandomForest",
            "data_type": "synthetic",
            "disclaimer": (
                "Educational prototype only. Not a real lending decision."
            ),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check backend logs."
        ) from error


from threading import Lock

from fastapi.concurrency import run_in_threadpool

_rag_pipeline = None
_rag_load_lock = Lock()
_rag_inference_lock = Lock()


def get_rag_pipeline():
    global _rag_pipeline

    if _rag_pipeline is None:
        with _rag_load_lock:
            if _rag_pipeline is None:
                # Import lazily because rag_pipeline.py loads
                # the embedding model and LLM during import.
                from src.llm import rag_pipeline
                _rag_pipeline = rag_pipeline

    return _rag_pipeline


def run_rag_analysis(application_data):
    pipeline = get_rag_pipeline()

    # Serialize inference for the initial local prototype.
    with _rag_inference_lock:
        return pipeline.analyze_application(application_data)


@app.post("/analyze")
async def analyze(application: LoanApplication):
    try:
        result = await run_in_threadpool(
            run_rag_analysis,
            application.model_dump()
        )

        return {
            "decision": str(result["prediction"]),
            "approval_probability": round(
                float(result["approval_probability"]), 4
            ),
            "similar_cases": result["similar_cases"],
            "counterfactuals": result["counterfactuals"],
            "ai_explanation": result["report"],
            "disclaimer": (
                "Educational prototype using synthetic data. "
                "Not a real lending decision."
            )
        }

    except Exception as error:
        import logging
        logging.exception("RAG analysis failed")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Check server logs."
        ) from error
