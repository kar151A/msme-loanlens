import pandas as pd


# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

df = pd.read_csv(
    "data/synthetic_applications.csv"
)


# ---------------------------------------------------------
# Basic Dataset Information
# ---------------------------------------------------------

print("\n==============================")
print("DATASET SHAPE")
print("==============================")
print(df.shape)


print("\n==============================")
print("FIRST 5 ROWS")
print("==============================")
print(df.head())


print("\n==============================")
print("COLUMN NAMES")
print("==============================")
print(df.columns.tolist())


print("\n==============================")
print("DATA TYPES")
print("==============================")
print(df.dtypes)


print("\n==============================")
print("MISSING VALUES")
print("==============================")
print(df.isnull().sum())


# ---------------------------------------------------------
# Decision Distribution
# ---------------------------------------------------------

print("\n==============================")
print("DECISION DISTRIBUTION")
print("==============================")
print(
    df["decision"].value_counts()
)


print("\n==============================")
print("DECISION PERCENTAGE")
print("==============================")

decision_percentage = (
    df["decision"]
    .value_counts(normalize=True)
    * 100
)

print(
    decision_percentage.round(2)
)


# ---------------------------------------------------------
# Basic Statistics
# ---------------------------------------------------------

print("\n==============================")
print("AVERAGE CIBIL SCORE")
print("==============================")
print(
    round(
        df["cibil_score"].mean(),
        2
    )
)


print("\n==============================")
print("AVERAGE ANNUAL REVENUE")
print("==============================")
print(
    round(
        df["annual_revenue_lakhs"].mean(),
        2
    )
)


print("\n==============================")
print("AVERAGE EXISTING DEBT")
print("==============================")
print(
    round(
        df["existing_debt_lakhs"].mean(),
        2
    )
)


print("\n==============================")
print("APPROVAL PERCENTAGE")
print("==============================")

approval_percentage = (
    (
        df["decision"]
        == "approved"
    ).mean()
    * 100
)

print(
    round(
        approval_percentage,
        2
    ),
    "%"
)


# ---------------------------------------------------------
# Compare Approved vs Rejected Applications
# ---------------------------------------------------------

print("\n==============================")
print("AVERAGE CIBIL BY DECISION")
print("==============================")

print(
    df.groupby(
        "decision"
    )["cibil_score"].mean().round(2)
)


print("\n==============================")
print("AVERAGE BUSINESS VINTAGE BY DECISION")
print("==============================")

print(
    df.groupby(
        "decision"
    )[
        "business_vintage_months"
    ].mean().round(2)
)


print("\n==============================")
print("AVERAGE DEBT RATIO BY DECISION")
print("==============================")

print(
    df.groupby(
        "decision"
    )[
        "debt_to_revenue_ratio"
    ].mean().round(2)
)


print("\n==============================")
print("AVERAGE LOAN RATIO BY DECISION")
print("==============================")

print(
    df.groupby(
        "decision"
    )[
        "loan_to_revenue_ratio"
    ].mean().round(2)
)


# ---------------------------------------------------------
# Sector Analysis
# ---------------------------------------------------------

print("\n==============================")
print("SECTOR DISTRIBUTION")
print("==============================")

print(
    df["sector"].value_counts()
)


print("\n==============================")
print("SECTOR PERCENTAGE")
print("==============================")

print(
    (
        df["sector"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


# ---------------------------------------------------------
# Rejection Reasons
# ---------------------------------------------------------

print("\n==============================")
print("TOP REJECTION REASONS")
print("==============================")

rejected_df = df[
    df["decision"] == "rejected"
]

print(
    rejected_df[
        "rejection_reasons"
    ]
    .value_counts()
    .head(10)
)


# ---------------------------------------------------------
# Numerical Summary
# ---------------------------------------------------------

print("\n==============================")
print("NUMERICAL SUMMARY")
print("==============================")

print(
    df.describe().round(2)
)