import faiss
import pandas as pd

from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INDEX_PATH = "models/msme_loan_faiss.index"

METADATA_PATH = "models/faiss_metadata.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Load FAISS Index
# ---------------------------------------------------------

index = faiss.read_index(
    INDEX_PATH
)


# ---------------------------------------------------------
# Load Metadata
# ---------------------------------------------------------

metadata = pd.read_csv(
    METADATA_PATH
)


# ---------------------------------------------------------
# Load Embedding Model
# ---------------------------------------------------------

embedding_model = SentenceTransformer(
    MODEL_NAME
)


# ---------------------------------------------------------
# Current Applicant
# ---------------------------------------------------------

application = {
    "business_vintage_months": 18,
    "annual_revenue_lakhs": 45.0,
    "existing_debt_lakhs": 12.0,
    "cibil_score": 720,
    "sector": "manufacturing",
    "loan_amount_lakhs": 15.0
}


# ---------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------

application[
    "debt_to_revenue_ratio"
] = round(
    application[
        "existing_debt_lakhs"
    ]
    / application[
        "annual_revenue_lakhs"
    ],
    2
)


application[
    "loan_to_revenue_ratio"
] = round(
    application[
        "loan_amount_lakhs"
    ]
    / application[
        "annual_revenue_lakhs"
    ],
    2
)


# ---------------------------------------------------------
# Convert Query To Text
# ---------------------------------------------------------

query_text = (
    f"Business vintage "
    f"{application['business_vintage_months']} months. "

    f"Annual revenue "
    f"{application['annual_revenue_lakhs']} lakh rupees. "

    f"Existing debt "
    f"{application['existing_debt_lakhs']} lakh rupees. "

    f"CIBIL score "
    f"{application['cibil_score']}. "

    f"Sector "
    f"{application['sector']}. "

    f"Requested loan "
    f"{application['loan_amount_lakhs']} lakh rupees. "

    f"Debt to revenue ratio "
    f"{application['debt_to_revenue_ratio']}. "

    f"Loan to revenue ratio "
    f"{application['loan_to_revenue_ratio']}."
)


print("\n==============================")
print("CURRENT APPLICATION")
print("==============================")

print(query_text)


# ---------------------------------------------------------
# Create Query Embedding
# ---------------------------------------------------------

query_embedding = embedding_model.encode(
    [query_text],
    convert_to_numpy=True
).astype(
    "float32"
)


faiss.normalize_L2(
    query_embedding
)


# ---------------------------------------------------------
# Search Similar Cases
# ---------------------------------------------------------

TOP_K = 5


similarities, indices = index.search(
    query_embedding,
    TOP_K
)


# ---------------------------------------------------------
# Display Results
# ---------------------------------------------------------

print("\n==============================")
print("TOP SIMILAR HISTORICAL CASES")
print("==============================")


for rank, (
    similarity,
    row_index
) in enumerate(
    zip(
        similarities[0],
        indices[0]
    ),
    start=1
):

    case = metadata.iloc[
        row_index
    ]

    print(
        f"\n---------- CASE {rank} ----------"
    )

    print(
        "Application ID:",
        case["application_id"]
    )

    print(
        "Similarity:",
        f"{similarity * 100:.2f}%"
    )

    print(
        "Decision:",
        case["decision"]
    )

    print(
        "CIBIL:",
        case["cibil_score"]
    )

    print(
        "Business Vintage:",
        case["business_vintage_months"],
        "months"
    )

    print(
        "Revenue:",
        f"₹{case['annual_revenue_lakhs']}L"
    )

    print(
        "Debt:",
        f"₹{case['existing_debt_lakhs']}L"
    )

    print(
        "Loan Requested:",
        f"₹{case['loan_amount_lakhs']}L"
    )

    print(
        "Sector:",
        case["sector"]
    )

    print(
        "Rejection Reasons:",
        case["rejection_reasons"]
    )

    print(
        "Officer Notes:",
        case["loan_officer_notes"]
    )