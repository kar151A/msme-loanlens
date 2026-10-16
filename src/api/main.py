
# ============================================================
# MSME LOANLENS - FASTAPI BACKEND
# ============================================================

import logging
import math
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("loanlens")


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "loan_model.pkl"


# ============================================================
# REQUEST VALIDATION
# ============================================================

class LoanApplication(BaseModel):

    model_config = ConfigDict(extra="forbid")

    business_vintage_months: int = Field(
        ge=0,
        le=600
    )

    annual_revenue_lakhs: float = Field(
        gt=0,
        allow_inf_nan=False
    )

    existing_debt_lakhs: float = Field(
        ge=0,
        allow_inf_nan=False
    )

    cibil_score: int = Field(
        ge=300,
        le=900
    )

    sector: str = Field(
        pattern="^(manufacturing|services|trading|agriculture)$"
    )

    loan_amount_lakhs: float = Field(
        gt=0,
        allow_inf_nan=False
    )


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

@lru_cache(maxsize=1)
def get_model():

    logger.info("Loading Random Forest model")

    if not MODEL_PATH.is_file():

        logger.error(
            "Model file not found: %s",
            MODEL_PATH
        )

        raise FileNotFoundError(
            "Trained model missing. "
            "Run src/ml/train_model.py first."
        )

    model = joblib.load(MODEL_PATH)

    logger.info("Random Forest model loaded successfully")

    return model


# ============================================================
# APPLICATION STARTUP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting MSME LoanLens API")

    get_model()

    logger.info("LoanLens API ready")

    yield

    logger.info("Shutting down LoanLens API")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="MSME LoanLens API",
    description=(
        "Explainable MSME loan prediction and "
        "RAG analysis prototype"
    ),
    version="0.2.0",
    lifespan=lifespan
)


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "MSME LoanLens",
        "version": "0.2.0"
    }


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def prepare_features(application_data):

    features = application_data.copy()

    revenue = features["annual_revenue_lakhs"]

    features["debt_to_revenue_ratio"] = round(
        features["existing_debt_lakhs"] / revenue,
        2
    )

    features["loan_to_revenue_ratio"] = round(
        features["loan_amount_lakhs"] / revenue,
        2
    )

    return features


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(application: LoanApplication):

    logger.info("Received loan prediction request")

    try:

        model = get_model()

        features = prepare_features(
            application.model_dump()
        )

        data = pd.DataFrame([features])

        prediction = str(
            model.predict(data)[0]
        )

        probabilities = model.predict_proba(data)[0]

        classes = list(model.classes_)

        approval_index = classes.index("approved")

        approval_probability = float(
            probabilities[approval_index]
        )

        logger.info(
            "Prediction completed: %s",
            prediction
        )

        return {
            "decision": prediction,

            "approval_probability": round(
                approval_probability,
                4
            ),

            "model_type": "RandomForest",

            "data_type": "synthetic",

            "disclaimer": (
                "Educational prototype only. "
                "Not a real lending decision."
            )
        }

    except Exception as error:

        logger.exception(
            "Prediction failed: %s",
            error
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check server logs."
        ) from error


# ============================================================
# RAG PIPELINE CONFIGURATION
# ============================================================

_rag_pipeline = None

_rag_load_lock = Lock()

_rag_inference_lock = Lock()


# ============================================================
# LOAD RAG PIPELINE
# ============================================================

def get_rag_pipeline():

    global _rag_pipeline

    if _rag_pipeline is None:

        with _rag_load_lock:

            if _rag_pipeline is None:

                logger.info(
                    "Loading RAG pipeline and language model"
                )

                from src.llm import rag_pipeline

                _rag_pipeline = rag_pipeline

                logger.info(
                    "RAG pipeline loaded successfully"
                )

    return _rag_pipeline


# ============================================================
# RUN RAG ANALYSIS
# ============================================================

def run_rag_analysis(application_data):

    logger.info("Preparing RAG analysis")

    pipeline = get_rag_pipeline()

    with _rag_inference_lock:

        logger.info("Starting ML + FAISS + LLM analysis")

        result = pipeline.analyze_application(
            application_data
        )

        logger.info("RAG analysis completed successfully")

        return result


# ============================================================
# JSON SERIALIZATION HELPER
# ============================================================

def convert_numpy(obj):

    """
    Recursively converts NumPy and Pandas values
    into JSON-compatible Python data types.
    """

    if isinstance(obj, np.ndarray):

        return convert_numpy(
            obj.tolist()
        )

    if isinstance(obj, np.generic):

        return convert_numpy(
            obj.item()
        )

    if obj is None:

        return None

    if obj is pd.NA or obj is pd.NaT:

        return None

    if isinstance(obj, dict):

        return {
            str(key): convert_numpy(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):

        return [
            convert_numpy(item)
            for item in obj
        ]

    if isinstance(obj, pd.Timestamp):

        return obj.isoformat()

    if isinstance(obj, float):

        if not math.isfinite(obj):

            return None

        return obj

    if isinstance(obj, (str, int, bool)):

        return obj

    raise TypeError(
        f"Unsupported response data type: {type(obj).__name__}"
    )


# ============================================================
# COMPLETE RAG ANALYSIS ENDPOINT
# ============================================================

@app.post("/analyze")
async def analyze(application: LoanApplication):

    logger.info("Received new loan analysis request")

    try:

        logger.info("Starting RAG analysis")

        result = await run_in_threadpool(
            run_rag_analysis,
            application.model_dump()
        )

        logger.info(
            "RAG pipeline returned results successfully"
        )

        # ----------------------------------------------------
        # BUILD API RESPONSE
        # ----------------------------------------------------

        response = {

            "decision": str(
                result["prediction"]
            ),

            "approval_probability": round(
                float(result["approval_probability"]),
                4
            ),

            "similar_cases": result["similar_cases"],

            "counterfactuals": result["counterfactuals"],

            "ai_explanation": result["report"],

            "disclaimer": (
                "Educational prototype using synthetic data. "
                "Not a real lending decision."
            )
        }

        logger.info(
            "Converting response to JSON-compatible format"
        )

        # ----------------------------------------------------
        # CONVERT NUMPY VALUES
        # ----------------------------------------------------

        clean_response = convert_numpy(
            response
        )

        logger.info(
            "JSON serialization preparation completed"
        )

        return clean_response

    except Exception as error:

        logger.exception(
            "LoanLens analysis failed: %s",
            error
        )

        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Check server logs."
        ) from error


# ============================================================
# END OF FILE
# ============================================================
