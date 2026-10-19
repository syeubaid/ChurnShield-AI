import csv
import random
import math

random.seed(42)

def generate_customer_data(n=500):
    rows = []
    for i in range(1, n+1):
        customer_id = f"CUST{i:04d}"
        age = random.randint(18, 75)
        gender = random.choice(["Male", "Female"])
        tenure_months = random.randint(1, 72)
        contract_type = random.choices(["Month-to-Month", "One Year", "Two Year"], weights=[0.5, 0.3, 0.2])[0]
        monthly_charges = round(random.uniform(20, 120), 2)
        total_charges = round(monthly_charges * tenure_months + random.uniform(-50, 50), 2)
        total_charges = max(0, total_charges)
        num_products = random.randint(1, 5)
        has_internet = random.choice(["Yes", "No"])
        internet_type = random.choices(["Fiber Optic", "DSL", "No Internet"], weights=[0.4, 0.3, 0.3])[0]
        online_security = random.choices(["Yes", "No", "No internet service"], weights=[0.4, 0.4, 0.2])[0]
        tech_support = random.choices(["Yes", "No", "No internet service"], weights=[0.35, 0.45, 0.2])[0]
        streaming_tv = random.choice(["Yes", "No"])
        streaming_movies = random.choice(["Yes", "No"])
        payment_method = random.choice(["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"])
        paperless_billing = random.choice(["Yes", "No"])
        num_support_calls = random.randint(0, 10)
        avg_call_duration = round(random.uniform(1, 30), 1)
        satisfaction_score = random.randint(1, 5)
        last_interaction_days = random.randint(1, 365)
        referrals_given = random.randint(0, 5)
        promotion_offered = random.choice(["Yes", "No"])
        late_payments = random.randint(0, 6)
        device_count = random.randint(1, 5)
        data_usage_gb = round(random.uniform(0, 100), 1)
        plan_upgrades = random.randint(0, 3)
        plan_downgrades = random.randint(0, 3)
        # Churn logic
        churn_score = 0
        if contract_type == "Month-to-Month": churn_score += 0.3
        if tenure_months < 12: churn_score += 0.2
        if satisfaction_score <= 2: churn_score += 0.3
        if num_support_calls >= 5: churn_score += 0.15
        if late_payments >= 3: churn_score += 0.2
        if monthly_charges > 90: churn_score += 0.1
        if referrals_given > 2: churn_score -= 0.15
        if plan_upgrades > 1: churn_score -= 0.1
        churn = 1 if (churn_score + random.uniform(-0.2, 0.2)) > 0.4 else 0
        rows.append([
            customer_id, age, gender, tenure_months, contract_type,
            monthly_charges, total_charges, num_products,
            has_internet, internet_type, online_security, tech_support,
            streaming_tv, streaming_movies, payment_method, paperless_billing,
            num_support_calls, avg_call_duration, satisfaction_score,
            last_interaction_days, referrals_given, promotion_offered,
            late_payments, device_count, data_usage_gb,
            plan_upgrades, plan_downgrades, churn
        ])
    return rows

headers = [
    "CustomerID", "Age", "Gender", "TenureMonths", "ContractType",
    "MonthlyCharges", "TotalCharges", "NumProducts",
    "HasInternet", "InternetType", "OnlineSecurity", "TechSupport",
    "StreamingTV", "StreamingMovies", "PaymentMethod", "PaperlessBilling",
    "NumSupportCalls", "AvgCallDurationMin", "SatisfactionScore",
    "LastInteractionDays", "ReferralsGiven", "PromotionOffered",
    "LatePayments", "DeviceCount", "DataUsageGB",
    "PlanUpgrades", "PlanDowngrades", "Churn"
]

data = generate_customer_data(500)

with open("/home/syedubaidullah/Desktop/customer churn system/customer_churn_dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(data)

print(f"Generated {len(data)} rows")
churn_count = sum(r[-1] for r in data)
print(f"Churn: {churn_count} ({churn_count/len(data)*100:.1f}%) | Non-churn: {len(data)-churn_count}")
