# LoanTap — Personal Loan Credit

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asmi2604/Loan-feature-engineering/blob/main/delhivery_analysis.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.13%2B-4C72B0.svg)](https://seaborn.pydata.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14%2B-8CAAE6.svg)](https://www.statsmodels.org/)

End-to-end **credit underwriting and binary classification** pipeline for LoanTap's personal loan portfolio.
Transforms raw loan application attributes into a scored underwriting model — performing EDA on 396K+ borrower records, engineering risk flags, building a calibrated Logistic Regression classifier, and surfacing actionable credit decision rules with threshold-tuned Precision–Recall tradeoffs.

---

## Highlights

| Feature | Detail |
| --- | --- |
| **Large-scale EDA** | 396,030 borrower records × 27 attributes — full univariate + bivariate analysis |
| **Risk flag engineering** | Binary flags for `pub_rec`, `mort_acc`, `pub_rec_bankruptcies` (presence > magnitude) |
| **Ordinal encoding** | Grade A→G mapped to 1–7 preserving natural credit risk ordering |
| **Missing value strategy** | Grouped median imputation for `mort_acc` by `total_acc`; mode fill for `emp_length` |
| **Outlier treatment** | 99th-percentile capping on `annual_inc`, `revol_bal`, `open_acc`, `total_acc` |
| **Class imbalance handling** | `class_weight='balanced'` in Logistic Regression — 80:20 Fully Paid vs Charged Off |
| **Threshold tuning** | Two operating scenarios — Balanced (F1-optimal) vs NPA-Safe (Recall ≥ 80%) |
| **Geographic analysis** | State-level default rates extracted from address field — 14% to 27% range |

---

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │             Raw Data Layer                  │
                    │  logistic_regression.csv                    │
                    │  396,030 rows × 27 columns                  │
                    │  Loan application + credit history records  │
                    └─────────────────┬───────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────────┐
                    │       Exploratory Data Analysis             │
                    │  Target distribution · Univariate analysis  │
                    │  Bivariate plots · Correlation heatmap      │
                    │  Geographic default rate mapping            │
                    └─────────────────┬───────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────────┐
                    │          Feature Engineering                │
                    │  pub_rec_flag · mort_acc_flag               │
                    │  pub_rec_bankruptcies_flag                  │
                    │  Grade ordinal · Term numeric               │
                    │  emp_length numeric · OHE categoricals      │
                    └──────────┬──────────────────┬──────────────┘
                               │                  │
          ┌────────────────────▼────┐   ┌─────────▼────────────────────┐
          │  Missing Value Treatment│   │   Outlier Treatment          │
          │  emp_length → mode      │   │   99th pct cap:              │
          │  revol_util → median    │   │   annual_inc, revol_bal      │
          │  mort_acc → grouped med │   │   open_acc, total_acc        │
          │  pub_rec_bk → 0         │   │                              │
          └────────────────────┬────┘   └─────────┬────────────────────┘
                               └────────┬─────────┘
                                        │
                    ┌───────────────────▼─────────────────────────┐
                    │        Data Preparation                     │
                    │  Drop: installment (r=0.95 with loan_amnt)  │
                    │  Stratified 80/20 train-test split          │
                    │  MinMaxScaler → features scaled to [0, 1]   │
                    └───────────────────┬─────────────────────────┘
                                        │
                    ┌───────────────────▼─────────────────────────┐
                    │        Logistic Regression Model            │
                    │  sklearn · class_weight=balanced            │
                    │  statsmodels · coefficient p-values         │
                    │  ROC-AUC = 0.70 · AP = 0.36                 │
                    └───────────────────┬─────────────────────────┘
                                        │
               ┌────────────────────────┼────────────────────────┐
               │                        │                        │
    ┌──────────▼──────────┐  ┌──────────▼──────────┐  ┌─────────▼──────────────┐
    │  Threshold Tuning   │  │  Model Evaluation   │  │  Business Framework    │
    │  Best F1 → t=0.52   │  │  ROC-AUC Curve      │  │  Auto-Approve tier     │
    │  NPA-Safe → t=0.41  │  │  PR Curve           │  │  Manual Review tier    │
    │  Scenario A vs B    │  │  Confusion Matrix   │  │  Auto-Reject tier      │
    └─────────────────────┘  └─────────────────────┘  └────────────────────────┘
```

---

## Project Structure

```
loantap-credit-underwriting/
│
├── LoanTap_CreditUnderwriting_Final.ipynb   ← Main notebook (open in Colab)
├── logistic_regression.csv                  ← Raw dataset (396,030 rows × 27 cols)
└── README.md                                ← This file
```

---

## Dataset

| Column | Description | Type |
| --- | --- | --- |
| `loan_amnt` | Listed loan amount applied by borrower | float64 |
| `term` | Repayment term — 36 or 60 months | object |
| `int_rate` | Interest rate on the loan (%) | float64 |
| `installment` | Monthly payment if loan originates | float64 |
| `grade` | LoanTap assigned credit grade (A–G) | object |
| `sub_grade` | LoanTap assigned credit sub-grade | object |
| `emp_title` | Borrower's job title | object |
| `emp_length` | Employment length in years (0–10+) | object |
| `home_ownership` | Home ownership status (RENT / MORTGAGE / OWN) | object |
| `annual_inc` | Self-reported annual income | float64 |
| `verification_status` | Income verification status | object |
| `issue_d` | Month the loan was funded | object |
| `loan_status` | **Target** — Fully Paid or Charged Off | object |
| `purpose` | Borrower-provided loan purpose category | object |
| `title` | Loan title provided by borrower | object |
| `dti` | Debt-to-income ratio (excl. mortgage) | float64 |
| `earliest_cr_line` | Month of earliest reported credit line | object |
| `open_acc` | Number of open credit lines | float64 |
| `pub_rec` | Number of derogatory public records | float64 |
| `revol_bal` | Total revolving credit balance | float64 |
| `revol_util` | Revolving line utilisation rate (%) | float64 |
| `total_acc` | Total number of credit lines | float64 |
| `initial_list_status` | Initial listing status — W or F | object |
| `application_type` | INDIVIDUAL or JOINT application | object |
| `mort_acc` | Number of mortgage accounts | float64 |
| `pub_rec_bankruptcies` | Number of public record bankruptcies | float64 |
| `address` | Borrower address (used for state extraction) | object |

- **396,030 rows** — large-scale real-world loan portfolio
- **Target imbalance:** 80.39% Fully Paid vs 19.61% Charged Off
- Missing values in: `mort_acc` (9.54%), `emp_title` (5.79%), `emp_length` (4.62%), `title` (0.44%), `pub_rec_bankruptcies` (0.14%), `revol_util` (0.07%)

---

## Notebook Walkthrough

| # | Section | What It Does |
| --- | --- | --- |
| 1 | Import Libraries | pandas, numpy, matplotlib, seaborn, sklearn, statsmodels — consistent plot styling |
| 2 | Data Loading & Inspection | Shape, dtypes, missing value bar chart, statistical summary |
| 3 | Target Variable Analysis | Count + pie chart of Fully Paid vs Charged Off — class imbalance discussion |
| 4 | Univariate — Numeric | Histograms + box plots for all 11 numeric features — skewness and outlier detection |
| 5 | Univariate — Categorical | Count plots for 8 categorical features + top 10 job titles bar chart |
| 6 | Bivariate Analysis | Loan status vs grade, term, loan amount, interest rate, home ownership, purpose, DTI, income |
| 7 | Correlation Heatmap | Lower-triangle heatmap — `loan_amnt` ↔ `installment` = 0.9539 multicollinearity flagged |
| 8 | Data Preprocessing | Dedup check → flags → missing treatment → outlier capping → encoding → scaling |
| 9 | Model Building | Logistic Regression (sklearn + statsmodels) — coefficient table + feature importance plot |
| 10 | Classification Report | Precision, Recall, F1 at default threshold 0.50 — confusion matrix |
| 11 | ROC-AUC Curve | AUC = 0.70 — curve with shaded area vs random classifier baseline |
| 12 | Precision-Recall Curve | AP = 0.36 — curve vs no-skill baseline — optimal operating region identified |
| 13 | Threshold Tuning | Precision / Recall / F1 vs threshold plot — Scenario A (t=0.52) vs Scenario B (t=0.41) |
| 14 | Geographic Analysis | Default rate by US state — NE/NV highest (~26%), IA/ME lowest (~14%) |
| 15 | Questionnaire Answers | All 9 assignment questions answered inline with evidence |
| 16 | Actionable Recommendations | 3-tier credit decision engine + 8 model deployment recommendations |

---

## Model Results Summary

### Default Threshold (t = 0.50)

| Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| Fully Paid | 0.88 | 0.67 | 0.76 | 63,671 |
| Charged Off | 0.31 | 0.63 | 0.42 | 15,535 |
| **Weighted Avg** | **0.77** | **0.66** | **0.69** | **79,206** |

### Scenario A — Balanced (t = 0.52, Best F1)

| Class | Precision | Recall | F1-Score |
| --- | --- | --- | --- |
| Fully Paid | 0.88 | 0.70 | 0.78 |
| Charged Off | 0.32 | 0.60 | 0.42 |

> Best for **normal market conditions** — maximises loan book size while keeping NPA manageable.

### Scenario B — NPA-Safe (t = 0.41, Recall ≥ 80%)

| Class | Precision | Recall | F1-Score |
| --- | --- | --- | --- |
| Fully Paid | 0.91 | 0.47 | 0.62 |
| Charged Off | 0.27 | 0.80 | 0.40 |

> Best for **economic stress or regulatory pressure** — aggressively catches 80% of all defaulters.

---

## Feature Engineering Summary

| Engineered Feature | Logic | Rationale |
| --- | --- | --- |
| `pub_rec_flag` | `pub_rec > 0 → 1` | Presence of any derogatory record matters more than the count |
| `mort_acc_flag` | `mort_acc > 0 → 1` | Having a mortgage signals financial commitment and creditworthiness |
| `pub_rec_bankruptcies_flag` | `pub_rec_bankruptcies > 0 → 1` | Any bankruptcy is a hard negative signal regardless of count |
| `grade` (ordinal) | A=1, B=2, …, G=7 | Preserves natural risk ordering — A is safest, G is riskiest |
| `term` (numeric) | Extract integer from "36 months" | Enables linear relationship modeling |
| `emp_length` (numeric) | `< 1 year`=0, …, `10+ years`=10 | Ordinal mapping preserves employment seniority signal |
| Drop `installment` | Corr = 0.9539 with `loan_amnt` | Eliminates multicollinearity — `loan_amnt` retained |
| State from `address` | Regex extract `[A-Z]{2}` | Enables geographic risk segmentation |

---

## Key Business Insights

| # | Insight | Business Impact |
| --- | --- | --- |
| 1 | **Grade is the #1 predictor** — Grade A defaults 6.3% vs Grade G defaults 38% | Grade A/B = fast-track approval; Grade E/F/G = reject or charge significant risk premium |
| 2 | **60-month term doubles default rate** vs 36-month (28% vs 16%) | Prefer 36-month term; require higher income proof for 60-month applications |
| 3 | **DTI > 25 strongly predicts default** | Mandatory additional income verification for DTI > 25 |
| 4 | **`small_business` purpose has highest default rate** across all loan purposes | Apply stricter underwriting criteria for business-purpose loans |
| 5 | **`revol_util` is the top coefficient feature** | Borrowers using > 80% of revolving credit are highest risk — add to rejection triggers |
| 6 | **Teacher and Manager are the safest job titles** (Top 2 by volume) | Stable salaried employees in education/management can be fast-tracked |
| 7 | **Geographic variation: 14% to 27% default rate** across US states | Apply state-level risk adjustment factor in pricing and caps |
| 8 | **80:20 class imbalance** — accuracy metric is misleading | Always evaluate using ROC-AUC and Recall; never use raw accuracy as KPI |

---

## Questionnaire Answers

| # | Question | Answer |
| --- | --- | --- |
| Q1 | % customers fully paid | **80.39%** |
| Q2 | Correlation: loan_amnt vs installment | **r = 0.9539** — very high positive; higher loan → higher installment |
| Q3 | Majority home ownership | **MORTGAGE** (50.1% of borrowers) |
| Q4 | Grade A fully pays more? | **TRUE** — 93.7% of Grade A borrowers fully pay |
| Q5 | Top 2 job titles | **Teacher** (4,389) and **Manager** (4,250) |
| Q6 | Primary bank metric | **Recall** (catch real defaulters); **ROC-AUC** for overall model selection |
| Q7 | Precision-Recall gap effect | High Recall + Low Precision = good borrowers rejected (lost revenue). High Precision + Low Recall = missed defaulters (NPAs) |
| Q8 | Top features | `grade`, `revol_util`, `annual_inc`, `dti`, `int_rate`, `pub_rec_flag`, `term`, `mort_acc_flag` |
| Q10 | Geographic effect | **YES** — state default rates range from ~14% (IA) to ~27% (NE) |

---

## Credit Decision Framework

```
          Loan Application Received
                    │
          ┌─────────▼──────────┐
          │  Model Scoring     │
          │  P(Charged Off)    │
          └─────────┬──────────┘
                    │
       ┌────────────┼────────────┐
       │            │            │
  P < 0.25     0.25–0.52    P > 0.52
       │            │            │
  ┌────▼────┐  ┌────▼────┐  ┌───▼─────┐
  │  AUTO   │  │ MANUAL  │  │  AUTO   │
  │ APPROVE │  │ REVIEW  │  │ REJECT  │
  └────┬────┘  └────┬────┘  └───┬─────┘
       │            │            │
  Grade A/B    Grade C/D    Grade E/F/G
  DTI < 15     DTI 15–25    DTI > 30
  No pub_rec   Unverified   pub_rec = 1
  36-mo term   Small biz    revol_util>85%
```

---

## Threshold Operating Modes

| Mode | Threshold | Use Case | Recall (Charged Off) | Precision (Charged Off) |
| --- | --- | --- | --- | --- |
| Default | 0.50 | Baseline | 0.63 | 0.31 |
| **Balanced (Scenario A)** | **0.52** | **Normal market** | **0.60** | **0.32** |
| **NPA-Safe (Scenario B)** | **0.41** | **Stress / regulatory** | **0.80** | **0.27** |

> Switch between modes based on quarterly NPA ratio review and RBI guidance.

---

## Actionable Recommendations

**1. Deploy Three-Tier Credit Engine**
Implement Auto-Approve / Manual Review / Auto-Reject tiers based on model probability scores. Target: 60% auto-approve, 25% manual review, 15% auto-reject to balance speed with risk control.

**2. Grade-Based Risk Pricing**
Set interest rate floors by grade: Grade A = base rate, each grade step adds 150–200bps. This prices NPA risk into the loan economics rather than binary approve/reject decisions.

**3. DTI Hard Cap**
Enforce a DTI ceiling of 35% for all personal loan applications. Applicants above this threshold should be auto-rejected regardless of other positive signals.

**4. Small Business Loan Separation**
Build a dedicated underwriting sub-model for `small_business` purpose loans — they have fundamentally different risk drivers than personal loans and should not share the same scoring model.

**5. State-Level Risk Adjustment**
Encode each state's historical default rate as a feature and apply a state risk multiplier to loan caps. High-default states (NE, NV, SD) should have 10–15% lower maximum approved loan amounts.

**6. Model Monitoring & Retraining**
Retrain the model monthly with new loan outcome data. Set a Recall alert threshold — if Recall for Charged Off drops below 70% in production, trigger an emergency retraining cycle.

**7. Feature Enrichment**
In the next model iteration, add: CIBIL/credit bureau score, employment type (government vs private), city tier (metro vs tier-2), and time-since-last-delinquency. Expected AUC lift: +0.05–0.10.

**8. Graduate to Gradient Boosting**
Logistic Regression (AUC=0.70) is a strong baseline. Upgrade to XGBoost or LightGBM for the production model — these handle non-linear interactions between grade, DTI, and income, typically achieving AUC of 0.76–0.82 on similar portfolios.

---

## Tech Stack

| Category | Technology |
| --- | --- |
| Language | Python 3.10 |
| Data manipulation | pandas · NumPy |
| Visualisation | Matplotlib · Seaborn |
| Machine learning | scikit-learn (LogisticRegression, MinMaxScaler, metrics) |
| Statistical modeling | statsmodels (Logit, summary2) |
| Notebook environment | Google Colab |

---

## Author

**Asmita Rajendra**

*Built as a credit risk case study — LoanTap Personal Loans, May 2026*
