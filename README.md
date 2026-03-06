# 💰 GenAI-Powered Financial Risk Analysis & Collections Strategy

An AI-driven financial risk analysis system that performs **Exploratory Data Analysis (EDA)** on customer financial data to identify delinquency risk indicators, builds predictive models, and generates intelligent collections strategies — with built-in **Ethical AI guardrails**.

---

## 🎯 What It Does

1. **EDA** — Analyzes key risk indicators: Credit Utilization, Missed Payments, Debt-to-Income Ratio
2. **Predictive Modeling** — Random Forest classifier to score each customer's delinquency risk
3. **Agentic AI Collections** — Generates SMART-goals-based intervention strategies per customer
4. **Ethical AI Guardrails** — Fairness checks to ensure no age/demographic bias in model outputs

---

## 🛠️ Tech Stack

| Area | Tools |
|------|-------|
| Language | Python |
| ML Framework | Scikit-learn |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| AI Approach | Generative AI, Predictive Modeling, EDA |

---

## 📊 Key Risk Indicators Analyzed

- 📈 **Credit Utilization** — % of available credit being used
- ❌ **Missed Payments** — Number of missed payments in last 6 months
- 💸 **Debt-to-Income Ratio** — Monthly debt vs. monthly income
- 🏦 **Credit Score** — Overall creditworthiness

---

## ⚙️ Getting Started

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/financial-risk-analysis.git
cd financial-risk-analysis

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python risk_analysis.py
```

---

## 🤖 Sample Output

```
📊 Delinquency Rate: 38.2%

📈 Key Risk Indicators:
  missed_payments_6m            corr=+0.487  🔴 HIGH RISK
  credit_utilization            corr=+0.401  🔴 HIGH RISK
  debt_to_income_ratio          corr=+0.312  🔴 HIGH RISK
  credit_score                  corr=-0.289  🟡 MODERATE

✅ ROC-AUC Score: 0.847

👤 Customer #42
   risk_tier: High Risk
   recommended_action: Proactive email campaign with flexible repayment options.
   smart_goal: Reduce balance by 25% within 60 days

ETHICAL AI GUARDRAILS — FAIRNESS CHECK
  Age 21-30: 38.5%  ✅ OK
  Age 31-40: 37.9%  ✅ OK
  ✅ Model passes fairness threshold (<10% disparity)
```

---

## 🛡️ Ethical AI Features

- **Fairness Auditing** — Checks for demographic bias across age groups
- **Explainability (XAI)** — Feature importance scores for every prediction
- **Regulatory Compliance** — Designed for responsible, auditable AI deployment

---

## 👤 Author

**Nuaman M** — [LinkedIn](https://linkedin.com/in/nuamanjaleel) | [GitHub](https://github.com/YOUR_USERNAME)
