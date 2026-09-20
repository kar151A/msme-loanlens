def get_positive_float(prompt):
    while True:
        try:
            value = float(input(prompt))

            if value <= 0:
                print("Value must be greater than 0.")
                continue

            return value

        except ValueError:
            print("Invalid input. Please enter a number.")


def get_non_negative_float(prompt):
    while True:
        try:
            value = float(input(prompt))

            if value < 0:
                print("Value cannot be negative.")
                continue

            return value

        except ValueError:
            print("Invalid input. Please enter a number.")


def get_business_vintage():
    while True:
        try:
            months = int(input("Business vintage (months): "))

            if months <= 0:
                print("Business vintage must be greater than 0.")
                continue

            return months

        except ValueError:
            print("Please enter a whole number.")


def get_cibil_score():
    while True:
        try:
            score = int(input("CIBIL score: "))

            if score < 300 or score > 900:
                print("CIBIL score must be between 300 and 900.")
                continue

            return score

        except ValueError:
            print("Please enter a valid integer.")


def get_sector():
    allowed_sectors = [
        "manufacturing",
        "services",
        "trading",
        "agriculture"
    ]

    while True:
        sector = input("Sector: ").strip().lower()

        if sector in allowed_sectors:
            return sector

        print(
            "Invalid sector. Choose from:",
            ", ".join(allowed_sectors)
        )


def calculate_debt_to_revenue(debt, revenue):
    return debt / revenue


print("================================")
print("          MSME LOANLENS")
print("================================")


business_vintage = get_business_vintage()

annual_revenue = get_positive_float(
    "Annual revenue (₹ lakhs): "
)

existing_debt = get_non_negative_float(
    "Existing debt (₹ lakhs): "
)

cibil_score = get_cibil_score()

sector = get_sector()

loan_amount = get_positive_float(
    "Requested loan amount (₹ lakhs): "
)


application = {
    "business_vintage_months": business_vintage,
    "annual_revenue_lakhs": annual_revenue,
    "existing_debt_lakhs": existing_debt,
    "cibil_score": cibil_score,
    "sector": sector,
    "loan_amount_lakhs": loan_amount
}


debt_ratio = calculate_debt_to_revenue(
    existing_debt,
    annual_revenue
)


print("\nApplication received:")
print(application)

print(
    "Debt-to-Revenue Ratio:",
    round(debt_ratio, 2)
)