import joblib
import pandas as pd


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

model = joblib.load(
    "models/loan_model.pkl"
)


# ---------------------------------------------------------
# Original Application
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

def prepare_application(app):

    app = app.copy()

    app["debt_to_revenue_ratio"] = round(
        app["existing_debt_lakhs"]
        / app["annual_revenue_lakhs"],
        2
    )

    app["loan_to_revenue_ratio"] = round(
        app["loan_amount_lakhs"]
        / app["annual_revenue_lakhs"],
        2
    )

    return app


# ---------------------------------------------------------
# Get approval probability
# ---------------------------------------------------------

def get_approval_probability(app):

    prepared_app = prepare_application(app)

    df = pd.DataFrame(
        [prepared_app]
    )

    probabilities = model.predict_proba(df)[0]

    classes = model.classes_

    approval_index = list(classes).index(
        "approved"
    )

    return probabilities[
        approval_index
    ]


# ---------------------------------------------------------
# Original probability
# ---------------------------------------------------------

original_probability = (
    get_approval_probability(application)
)


print("\n================================")
print("ORIGINAL APPLICATION")
print("================================")

for key, value in application.items():
    print(f"{key}: {value}")


print("\nCurrent Approval Probability:")

print(
    f"{original_probability * 100:.2f}%"
)


# ---------------------------------------------------------
# Generate Counterfactual Scenarios
# ---------------------------------------------------------

counterfactuals = []


# ---------------------------------------------------------
# Scenario 1: Improve CIBIL
# ---------------------------------------------------------

for new_cibil in [
    730,
    750,
    770,
    800
]:

    if new_cibil <= application["cibil_score"]:
        continue

    modified = application.copy()

    modified["cibil_score"] = new_cibil

    probability = (
        get_approval_probability(modified)
    )

    counterfactuals.append(
        {
            "change":
                f"Increase CIBIL to {new_cibil}",

            "new_probability":
                probability,

            "improvement":
                probability
                - original_probability
        }
    )


# ---------------------------------------------------------
# Scenario 2: Reduce Debt
# ---------------------------------------------------------

current_debt = application[
    "existing_debt_lakhs"
]


for reduction in [
    0.10,
    0.25,
    0.50,
    0.75
]:

    new_debt = (
        current_debt
        * (1 - reduction)
    )

    modified = application.copy()

    modified[
        "existing_debt_lakhs"
    ] = round(
        new_debt,
        2
    )

    probability = (
        get_approval_probability(modified)
    )

    counterfactuals.append(
        {
            "change":
                (
                    f"Reduce debt to "
                    f"₹{new_debt:.2f}L"
                ),

            "new_probability":
                probability,

            "improvement":
                probability
                - original_probability
        }
    )


# ---------------------------------------------------------
# Scenario 3: Increase Business Vintage
# ---------------------------------------------------------

for new_vintage in [
    24,
    30,
    36,
    48
]:

    if (
        new_vintage
        <= application[
            "business_vintage_months"
        ]
    ):
        continue

    modified = application.copy()

    modified[
        "business_vintage_months"
    ] = new_vintage

    probability = (
        get_approval_probability(modified)
    )

    counterfactuals.append(
        {
            "change":
                (
                    f"Increase business "
                    f"vintage to "
                    f"{new_vintage} months"
                ),

            "new_probability":
                probability,

            "improvement":
                probability
                - original_probability
        }
    )


# ---------------------------------------------------------
# Scenario 4: Reduce Requested Loan
# ---------------------------------------------------------

current_loan = application[
    "loan_amount_lakhs"
]


for reduction in [
    0.10,
    0.25,
    0.50
]:

    new_loan = (
        current_loan
        * (1 - reduction)
    )

    modified = application.copy()

    modified[
        "loan_amount_lakhs"
    ] = round(
        new_loan,
        2
    )

    probability = (
        get_approval_probability(modified)
    )

    counterfactuals.append(
        {
            "change":
                (
                    f"Reduce requested loan "
                    f"to ₹{new_loan:.2f}L"
                ),

            "new_probability":
                probability,

            "improvement":
                probability
                - original_probability
        }
    )


# ---------------------------------------------------------
# Rank recommendations
# ---------------------------------------------------------

counterfactuals.sort(
    key=lambda x: x["improvement"],
    reverse=True
)


# ---------------------------------------------------------
# Display best recommendations
# ---------------------------------------------------------

print("\n================================")
print("TOP COUNTERFACTUAL RECOMMENDATIONS")
print("================================")


for number, result in enumerate(
    counterfactuals[:5],
    start=1
):

    print(
        f"\n{number}. {result['change']}"
    )

    print(
        "   New Approval Probability:",
        f"{result['new_probability'] * 100:.2f}%"
    )

    print(
        "   Improvement:",
        f"+{result['improvement'] * 100:.2f}%"
    )