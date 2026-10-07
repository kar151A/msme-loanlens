import os

import faiss
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = "data/synthetic_applications.csv"

INDEX_PATH = "models/msme_loan_faiss.index"

METADATA_PATH = "models/faiss_metadata.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

print("\n==============================")
print("LOADING DATASET")
print("==============================")

df = pd.read_csv(DATA_PATH)

print("Applications loaded:", len(df))


# ---------------------------------------------------------
# Convert Application Row To Text
# ---------------------------------------------------------

def application_to_text(row):

    return (
        f"Business vintage {row['business_vintage_months']} months. "
        f"Annual revenue {row['annual_revenue_lakhs']} lakh rupees. "
        f"Existing debt {row['existing_debt_lakhs']} lakh rupees. "
        f"CIBIL score {row['cibil_score']}. "
        f"Sector {row['sector']}. "
        f"Requested loan {row['loan_amount_lakhs']} lakh rupees. "
        f"Debt to revenue ratio {row['debt_to_revenue_ratio']}. "
        f"Loan to revenue ratio {row['loan_to_revenue_ratio']}."
    )


df["search_text"] = df.apply(
    application_to_text,
    axis=1
)


print("\nExample document:")
print(df["search_text"].iloc[0])


# ---------------------------------------------------------
# Load Embedding Model
# ---------------------------------------------------------

print("\n==============================")
print("LOADING EMBEDDING MODEL")
print("==============================")

embedding_model = SentenceTransformer(
    MODEL_NAME
)


# ---------------------------------------------------------
# Generate Embeddings
# ---------------------------------------------------------

print("\n==============================")
print("GENERATING EMBEDDINGS")
print("==============================")

embeddings = embedding_model.encode(
    df["search_text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)


embeddings = embeddings.astype(
    "float32"
)


print(
    "Embedding shape:",
    embeddings.shape
)


# ---------------------------------------------------------
# Normalize Embeddings
# ---------------------------------------------------------

faiss.normalize_L2(
    embeddings
)


# ---------------------------------------------------------
# Create FAISS Index
# ---------------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)


# ---------------------------------------------------------
# Add Embeddings To Index
# ---------------------------------------------------------

index.add(
    embeddings
)


print(
    "Vectors stored:",
    index.ntotal
)


# ---------------------------------------------------------
# Create Models Folder
# ---------------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)


# ---------------------------------------------------------
# Save FAISS Index
# ---------------------------------------------------------

faiss.write_index(
    index,
    INDEX_PATH
)


# ---------------------------------------------------------
# Save Metadata
# ---------------------------------------------------------

metadata_columns = [
    "application_id",
    "business_vintage_months",
    "annual_revenue_lakhs",
    "existing_debt_lakhs",
    "cibil_score",
    "sector",
    "loan_amount_lakhs",
    "debt_to_revenue_ratio",
    "loan_to_revenue_ratio",
    "decision",
    "rejection_reasons",
    "loan_officer_notes",
    "search_text"
]


df[
    metadata_columns
].to_csv(
    METADATA_PATH,
    index=False
)


# ---------------------------------------------------------
# Final Output
# ---------------------------------------------------------

print("\n==============================")
print("FAISS INDEX BUILT SUCCESSFULLY")
print("==============================")

print(
    "Index path:",
    INDEX_PATH
)

print(
    "Metadata path:",
    METADATA_PATH
)

print(
    "Embedding dimension:",
    dimension
)

print(
    "Applications indexed:",
    index.ntotal
)