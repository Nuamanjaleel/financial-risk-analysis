"""
GenAI-Powered Financial Risk Analysis & Collections Strategy
Author: Nuaman M

Performs Exploratory Data Analysis (EDA) on financial datasets to identify
delinquency risk indicators, build predictive models, and generate
AI-powered collections intervention strategies with Ethical AI guardrails.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────

def generate_financial_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """Generate a synthetic financial dataset for demonstration."""
    np.random.seed(42)

    data = pd.DataFrame({
        "customer_id": range(1, n_samples + 1),
        "credit_utilization": np.random.uniform(0, 1, n_samples),        # 0–100%
        "missed_payments_6m": np.random.randint(0, 6, n_samples),         # Last 6 months
        "debt_to_income_ratio": np.random.uniform(0.1, 0.9, n_samples),  # DTI ratio
        "credit_score": np.random.randint(300, 850, n_samples),
        "loan_amount": np.random.randint(1000, 50000, n_samples),
        "employment_months": np.random.randint(0, 120, n_samples),
        "age": np.random.randint(21, 70, n_samples),
    })

    # Delinquency label: 1 = at risk, 0 = safe
    risk_score = (
        0.4 * data["credit_utilization"] +
        0.3 * (data["missed_payments_6m"] / 6) +
        0.2 * data["debt_to_income_ratio"] +
        0.1 * (1 - data["credit_score"] / 850)
    )
    data["is_delinquent"] = (risk_score > 0.4).astype(int)
    return data


# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────

def run_eda(df: pd.DataFrame):
    """Run EDA to identify key delinquency risk indicators."""
    print("\n" + "=" * 55)
    print("   EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 55)
    print(f"\n📊 Dataset Shape: {df.shape}")
    print(f"⚠️  Delinquency Rate: {df['is_delinquent'].mean():.1%}")

    print("\n📈 Key Risk Indicators (correlation with delinquency):")
    features = ["credit_utilization", "missed_payments_6m",
                "debt_to_income_ratio", "credit_score"]
    for feat in features:
        corr = df[feat].corr(df["is_delinquent"])
        direction = "🔴 HIGH RISK" if abs(corr) > 0.3 else "🟡 MODERATE"
        print(f"  {feat:<30} corr={corr:+.3f}  {direction}")

    print("\n📋 Summary Statistics:")
    print(df[features].describe().round(2).to_string())


# ─────────────────────────────────────────────
# 3. PREDICTIVE MODELING
# ─────────────────────────────────────────────

def train_risk_model(df: pd.DataFrame):
    """Train a Random Forest classifier for delinquency prediction."""
    print("\n" + "=" * 55)
    print("   PREDICTIVE RISK MODEL (Random Forest)")
    print("=" * 55)

    features = ["credit_utilization", "missed_payments_6m",
                "debt_to_income_ratio", "credit_score",
                "loan_amount", "employment_months"]
    X = df[features]
    y = df["is_delinquent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    print(f"\n✅ ROC-AUC Score: {roc_auc_score(y_test, y_prob):.3f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Safe", "At Risk"]))

    # Feature importance
    importances = dict(zip(features, model.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print("🔍 Feature Importance:")
    for feat, imp in sorted_imp:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<30} {bar} {imp:.3f}")

    return model, scaler


# ─────────────────────────────────────────────
# 4. AGENTIC AI COLLECTIONS STRATEGY
# ─────────────────────────────────────────────

def generate_collections_strategy(customer: dict, risk_score: float) -> dict:
    """Generate a SMART-goals-based collections strategy using AI logic."""
    if risk_score >= 0.75:
        tier = "Critical"
        action = "Immediate outreach via phone + email. Offer hardship payment plan."
        goal = "Prevent charge-off within 30 days"
    elif risk_score >= 0.50:
        tier = "High Risk"
        action = "Proactive email campaign with flexible repayment options."
        goal = "Reduce balance by 25% within 60 days"
    elif risk_score >= 0.30:
        tier = "Moderate Risk"
        action = "Automated reminder SMS + self-service portal nudge."
        goal = "Maintain on-time payments for next 3 months"
    else:
        tier = "Low Risk"
        action = "Routine monitoring. No intervention required."
        goal = "Continue healthy repayment behavior"

    return {
        "customer_id": customer.get("customer_id"),
        "risk_tier": tier,
        "risk_score": round(risk_score, 3),
        "recommended_action": action,
        "smart_goal": goal
    }


# ─────────────────────────────────────────────
# 5. ETHICAL AI GUARDRAILS
# ─────────────────────────────────────────────

def fairness_check(df: pd.DataFrame):
    """
    Ethical AI: Check that protected attributes (age) don't
    disproportionately influence risk classification.
    """
    print("\n" + "=" * 55)
    print("   ETHICAL AI GUARDRAILS — FAIRNESS CHECK")
    print("=" * 55)

    df["age_group"] = pd.cut(df["age"], bins=[20, 30, 40, 50, 70],
                              labels=["21-30", "31-40", "41-50", "51-70"])

    fairness = df.groupby("age_group")["is_delinquent"].mean()
    print("\n📊 Delinquency Rate by Age Group (should be similar):")
    for group, rate in fairness.items():
        flag = "⚠️  BIAS ALERT" if abs(rate - fairness.mean()) > 0.1 else "✅ OK"
        print(f"  Age {group}: {rate:.1%}  {flag}")

    max_disparity = fairness.max() - fairness.min()
    print(f"\n  Max Disparity: {max_disparity:.1%}")
    if max_disparity < 0.1:
        print("  ✅ Model passes fairness threshold (<10% disparity)")
    else:
        print("  ⚠️  Model requires fairness review before deployment")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀 GenAI Financial Risk Analysis System — Starting...\n")

    # Step 1: Load/generate data
    df = generate_financial_dataset(1000)

    # Step 2: EDA
    run_eda(df)

    # Step 3: Train model
    model, scaler = train_risk_model(df)

    # Step 4: Demo collections strategy
    print("\n" + "=" * 55)
    print("   AGENTIC AI COLLECTIONS STRATEGY (DEMO)")
    print("=" * 55)
    sample_customers = df.sample(3, random_state=1).to_dict("records")
    features = ["credit_utilization", "missed_payments_6m",
                "debt_to_income_ratio", "credit_score",
                "loan_amount", "employment_months"]
    for customer in sample_customers:
        X = np.array([[customer[f] for f in features]])
        X_scaled = scaler.transform(X)
        risk = model.predict_proba(X_scaled)[0][1]
        strategy = generate_collections_strategy(customer, risk)
        print(f"\n👤 Customer #{strategy['customer_id']}")
        for k, v in strategy.items():
            if k != "customer_id":
                print(f"   {k}: {v}")

    # Step 5: Ethical AI check
    fairness_check(df)

    print("\n✅ Analysis Complete.\n")
