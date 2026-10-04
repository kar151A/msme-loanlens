import joblib
import pandas as pd


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

pipeline = joblib.load(
    "models/loan_model.pkl"
)


# ---------------------------------------------------------
# Access preprocessing + classifier
# ---------------------------------------------------------

preprocessor = pipeline.named_steps[
    "preprocessor"
]

classifier = pipeline.named_steps[
    "classifier"
]


# ---------------------------------------------------------
# Get transformed feature names
# ---------------------------------------------------------

feature_names = (
    preprocessor.get_feature_names_out()
)


# ---------------------------------------------------------
# Get feature importance
# ---------------------------------------------------------

importances = classifier.feature_importances_


importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": importances
    }
)


importance_df = (
    importance_df
    .sort_values(
        by="importance",
        ascending=False
    )
)


print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")

print(
    importance_df.head(15)
)


# ---------------------------------------------------------
# Create new applicant
# ---------------------------------------------------------

new_application = pd.DataFrame(
    [
        {
            "business_vintage_months": 18,
            "annual_revenue_lakhs": 45.0,
            "existing_debt_lakhs": 12.0,
            "cibil_score": 720,
            "sector": "manufacturing",
            "loan_amount_lakhs": 15.0,
            "debt_to_revenue_ratio": round(
                12.0 / 45.0,
                2
            ),
            "loan_to_revenue_ratio": round(
                15.0 / 45.0,
                2
            )
        }
    ]
)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

prediction = pipeline.predict(
    new_application
)[0]


probabilities = pipeline.predict_proba(
    new_application
)[0]


classes = pipeline.classes_


# ---------------------------------------------------------
# Show result
# ---------------------------------------------------------

print("\n==============================")
print("NEW APPLICATION")
print("==============================")

print(
    new_application.to_string(
        index=False
    )
)


print("\n==============================")
print("PREDICTED DECISION")
print("==============================")

print(
    prediction.upper()
)


print("\n==============================")
print("CLASS PROBABILITIES")
print("==============================")


for class_name, probability in zip(
    classes,
    probabilities
):

    print(
        f"{class_name}: "
        f"{probability * 100:.2f}%"
    )