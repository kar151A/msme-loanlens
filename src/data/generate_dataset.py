import random
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------
# Generate one synthetic MSME loan application
# ---------------------------------------------------------

def generate_application(application_id):

    # Business vintage:
    # Most businesses are around 3 years old,
    # but we allow a wider realistic range.
    business_vintage_months = int(
        np.clip(
            np.random.normal(
                loc=36,
                scale=24
            ),
            6,
            180
        )
    )


    # Annual revenue:
    # Log-normal distribution gives many small/medium businesses
    # and fewer very large businesses.
    annual_revenue_lakhs = round(
        float(
            np.clip(
                np.random.lognormal(
                    mean=3.7,
                    sigma=0.7
                ),
                5,
                300
            )
        ),
        2
    )


    # Existing debt:
    # Debt is related to revenue instead of being completely random.
    debt_ratio_generated = np.random.beta(
        a=2,
        b=5
    )

    existing_debt_lakhs = round(
        annual_revenue_lakhs
        * debt_ratio_generated,
        2
    )


    # CIBIL score:
    # Scores around 700-750 are more common.
    cibil_score = int(
        np.clip(
            np.random.normal(
                loc=720,
                scale=70
            ),
            300,
            900
        )
    )


    # Sector:
    # Different sectors have different probabilities.
    sector = random.choices(
        population=[
            "manufacturing",
            "services",
            "trading",
            "agriculture"
        ],
        weights=[
            30,
            35,
            25,
            10
        ],
        k=1
    )[0]


    # Loan amount:
    # Loan requested is partly related to annual revenue.
    loan_fraction = np.random.uniform(
        0.1,
        0.8
    )

    loan_amount_lakhs = round(
        annual_revenue_lakhs
        * loan_fraction,
        2
    )


    # ---------------------------------------------------------
    # Feature Engineering
    # ---------------------------------------------------------

    debt_to_revenue_ratio = (
        existing_debt_lakhs
        / annual_revenue_lakhs
    )

    loan_to_revenue_ratio = (
        loan_amount_lakhs
        / annual_revenue_lakhs
    )


    # ---------------------------------------------------------
    # Synthetic Loan Approval Score
    # ---------------------------------------------------------

    score = 0


    # CIBIL score contribution
    if cibil_score >= 750:
        score += 3

    elif cibil_score >= 700:
        score += 2

    elif cibil_score >= 650:
        score += 1


    # Business vintage contribution
    if business_vintage_months >= 36:
        score += 2

    elif business_vintage_months >= 24:
        score += 1


    # Debt ratio contribution
    if debt_to_revenue_ratio <= 0.25:
        score += 2

    elif debt_to_revenue_ratio <= 0.4:
        score += 1


    # Loan ratio contribution
    if loan_to_revenue_ratio <= 0.3:
        score += 2

    elif loan_to_revenue_ratio <= 0.5:
        score += 1


    # ---------------------------------------------------------
    # Final Decision
    # ---------------------------------------------------------

    if score >= 6:
        decision = "approved"

    else:
        decision = "rejected"


    # ---------------------------------------------------------
    # Rejection Reasons
    # ---------------------------------------------------------

    rejection_reasons = []


    if cibil_score < 650:
        rejection_reasons.append(
            "low_cibil_score"
        )


    if business_vintage_months < 24:
        rejection_reasons.append(
            "insufficient_business_vintage"
        )


    if debt_to_revenue_ratio > 0.4:
        rejection_reasons.append(
            "high_debt_to_revenue_ratio"
        )


    if loan_to_revenue_ratio > 0.5:
        rejection_reasons.append(
            "high_loan_to_revenue_ratio"
        )


    # Approved applications should not have rejection reasons.
    if decision == "approved":
        rejection_reasons = []


    # ---------------------------------------------------------
    # Loan Officer Notes
    # ---------------------------------------------------------

    if decision == "approved":

        loan_officer_notes = (
            "Application approved based on satisfactory "
            "credit profile, business vintage and debt levels."
        )

    else:

        if rejection_reasons:

            loan_officer_notes = (
                "Application rejected due to: "
                + ", ".join(rejection_reasons)
            )

        else:

            loan_officer_notes = (
                "Application rejected because the overall "
                "risk score did not meet the approval threshold."
            )


    # ---------------------------------------------------------
    # Return application as dictionary
    # ---------------------------------------------------------

    return {

        "application_id":
            application_id,

        "business_vintage_months":
            business_vintage_months,

        "annual_revenue_lakhs":
            annual_revenue_lakhs,

        "existing_debt_lakhs":
            existing_debt_lakhs,

        "cibil_score":
            cibil_score,

        "sector":
            sector,

        "loan_amount_lakhs":
            loan_amount_lakhs,

        "debt_to_revenue_ratio":
            round(
                debt_to_revenue_ratio,
                2
            ),

        "loan_to_revenue_ratio":
            round(
                loan_to_revenue_ratio,
                2
            ),

        "decision":
            decision,

        "rejection_reasons":
            "|".join(
                rejection_reasons
            ),

        "loan_officer_notes":
            loan_officer_notes
    }


# ---------------------------------------------------------
# Generate Dataset
# ---------------------------------------------------------

applications = []


for i in range(5000):

    application = generate_application(
        f"APP{i + 1:04d}"
    )

    applications.append(
        application
    )


# ---------------------------------------------------------
# Convert to Pandas DataFrame
# ---------------------------------------------------------

df = pd.DataFrame(
    applications
)


# ---------------------------------------------------------
# Preview Dataset
# ---------------------------------------------------------

print("\nFIRST 5 APPLICATIONS")

print(
    df.head()
)


# ---------------------------------------------------------
# Save Dataset
# ---------------------------------------------------------

df.to_csv(
    "data/synthetic_applications.csv",
    index=False
)


print("\nDataset generated successfully!")

print(
    "Total applications:",
    len(df)
)


print(
    "Saved to:",
    "data/synthetic_applications.csv"
)