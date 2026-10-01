import pandas as pd


df = pd.read_csv(
    "data/synthetic_applications.csv"
)


print("DATASET SHAPE")
print(df.shape)


print("\nFIRST 5 ROWS")
print(df.head())


print("\nCOLUMN NAMES")
print(df.columns.tolist())


print("\nDATA TYPES")
print(df.dtypes)


print("\nMISSING VALUES")
print(df.isnull().sum())


print("\nDECISION DISTRIBUTION")
print(df["decision"].value_counts())

print("\nNUMERICAL SUMMARY")
print(df.describe())

print("\nAVERAGE CIBIL SCORE")
print(round(df["cibil_score"].mean(), 2))

print("\nAVERAGE ANNUAL REVENUE")
print(round(df["annual_revenue_lakhs"].mean(), 2))

print("\nAVERAGE EXISTING DEBT")
print(round(df["existing_debt_lakhs"].mean(), 2))

approval_percentage = (
    (df["decision"] == "approved").mean() * 100
)

print("\nAPPROVAL PERCENTAGE")
print(round(approval_percentage, 2), "%")

print("\nAVERAGE CIBIL BY DECISION")
print(
    df.groupby("decision")["cibil_score"].mean()
)

print("\nAVERAGE DEBT RATIO BY DECISION")
print(
    df.groupby("decision")[
        "debt_to_revenue_ratio"
    ].mean()
)

print("\nAVERAGE VINTAGE BY DECISION")
print(
    df.groupby("decision")[
        "business_vintage_months"
    ].mean()
)