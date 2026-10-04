import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

df = pd.read_csv(
    "data/synthetic_applications.csv"
)


# ---------------------------------------------------------
# Select Features
# ---------------------------------------------------------

features = [
    "business_vintage_months",
    "annual_revenue_lakhs",
    "existing_debt_lakhs",
    "cibil_score",
    "sector",
    "loan_amount_lakhs",
    "debt_to_revenue_ratio",
    "loan_to_revenue_ratio"
]


target = "decision"


X = df[features]
y = df[target]


# ---------------------------------------------------------
# Train/Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\n==============================")
print("DATA SPLIT")
print("==============================")

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

categorical_features = [
    "sector"
]

numerical_features = [
    "business_vintage_months",
    "annual_revenue_lakhs",
    "existing_debt_lakhs",
    "cibil_score",
    "loan_amount_lakhs",
    "debt_to_revenue_ratio",
    "loan_to_revenue_ratio"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ---------------------------------------------------------
# Random Forest Model
# ---------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# ---------------------------------------------------------
# Full ML Pipeline
# ---------------------------------------------------------

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            model
        )
    ]
)


# ---------------------------------------------------------
# Train Model
# ---------------------------------------------------------

print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train
)

print("Training complete.")


# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------

y_pred = pipeline.predict(
    X_test
)


# ---------------------------------------------------------
# Accuracy
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n==============================")
print("MODEL ACCURACY")
print("==============================")

print(
    round(
        accuracy * 100,
        2
    ),
    "%"
)


# ---------------------------------------------------------
# Classification Report
# ---------------------------------------------------------

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ---------------------------------------------------------
# Save Model
# ---------------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(
    pipeline,
    "models/loan_model.pkl"
)


print("\n==============================")
print("MODEL SAVED")
print("==============================")

print(
    "models/loan_model.pkl"
)