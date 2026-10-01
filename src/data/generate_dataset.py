import random
import pandas as pd
random.seed(42)
def generate_application(application_id):
    business_vintage_months = random.randint(6, 120)

    
    annual_revenue_lakhs = round(
        random.uniform(10, 200),
        2
    )

    existing_debt_lakhs = round(
        random.uniform(0, 80),
        2
    )

    cibil_score = random.randint(300, 900)

    sector = random.choice([
        "manufacturing",
        "services",
        "trading",
        "agriculture"
    ])

    loan_amount_lakhs = round(
        random.uniform(5, 100),
        2
    )

    debt_to_revenue_ratio = (
            existing_debt_lakhs / annual_revenue_lakhs
        )

    if (cibil_score >= 700 and business_vintage_months >= 24 and debt_to_revenue_ratio <= 0.4):
        decision = "approved"
    else:
        decision = "rejected"
    
    return {
    "application_id": application_id,
    "business_vintage_months": business_vintage_months,
    "annual_revenue_lakhs": annual_revenue_lakhs,
    "existing_debt_lakhs": existing_debt_lakhs,
    "cibil_score": cibil_score,
    "sector": sector,
    "loan_amount_lakhs": loan_amount_lakhs,
    "debt_to_revenue_ratio": round(debt_to_revenue_ratio, 2),
    "decision": decision
}


applications = []

for i in range(5000):
    application = generate_application(
        f"APP{i + 1:04d}"
    )

    applications.append(application)




df = pd.DataFrame(applications)
print(df.head())

df.to_csv(
    "data/synthetic_applications.csv",
    index=False
)