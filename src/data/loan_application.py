business_vintage_months = 18
annual_revenue_lakhs = 45.0
existing_debt_lakhs = 12.0
cibil_score = 720
sector = "manufacturing"
loan_amount_lakhs = 15.0


loan_application = {
    "business_vintage_months": business_vintage_months,
    "annual_revenue_lakhs": annual_revenue_lakhs,
    "existing_debt_lakhs": existing_debt_lakhs,
    "cibil_score": cibil_score,
    "sector": sector,
    "loan_amount_lakhs": loan_amount_lakhs,
}


print("MSME LoanLens")
print("--------------------")

debt_to_revenue_ratio = (
    existing_debt_lakhs / annual_revenue_lakhs
)
loan_to_revenue_ratio = (
    loan_amount_lakhs / annual_revenue_lakhs
)
print(
    "Debt-to-Revenue Ratio:",
    round(debt_to_revenue_ratio, 2)
)

print(
    "Loan-to-Revenue Ratio:",
    round(loan_to_revenue_ratio, 2)
)
revenue_to_loan_ratio = (
    annual_revenue_lakhs / loan_amount_lakhs
)
print(
    "Revenue-to-Loan Ratio:",
    round(revenue_to_loan_ratio, 2)
)