# 📦 Vendor Invoice Intelligence System
**Freight Cost Prediction & Invoice Risk Flagging**

An end-to-end machine learning project that helps finance and procurement teams **estimate freight cost** for vendor invoices and **flag risky invoices** for manual review. Two models are trained on a SQLite inventory database and served through a Streamlit web app.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![SQLite](https://img.shields.io/badge/SQLite-Data-lightgrey)

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Business Objectives](#-business-objectives)
- [Dataset](#-dataset)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Models & Results](#-models--results)
- [Streamlit Application](#-streamlit-application)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Known Limitations & Future Work](#-known-limitations--future-work)
- [Author](#-author)

---

## 📌 Project Overview

This project delivers two machine learning modules for vendor invoice analytics:

1. **Freight Cost Prediction (Regression)** – predicts the expected freight charge of an invoice from its dollar value, supporting budgeting, forecasting and vendor negotiations.
2. **Invoice Manual-Approval Flagging (Classification)** – predicts whether an invoice should be routed for manual review because of abnormal cost or delivery patterns, reducing financial leakage and manual workload.

The workflow covers SQL-based data extraction, feature engineering, EDA with statistical testing, model comparison, hyperparameter tuning, model persistence, and a Streamlit front end.

---

## 🎯 Business Objectives

- **Forecast freight cost** accurately to improve budgeting and vendor negotiations.
- **Detect abnormal invoices** (value mismatches, slow receiving) before payment.
- **Reduce manual effort** by auto-approving low-risk invoices and focusing reviewers on risky ones.
- **Reduce financial leakage** from billing errors and anomalies.

---

## 🗂 Dataset

Data lives in a single SQLite database, `data/inventory.db` (~424 MB), covering purchase orders from **Dec 2023 – Dec 2024**.

| Table | Rows | Description |
|---|---:|---|
| `vendor_invoice` | 5,543 | One row per vendor invoice: vendor, PO/invoice/pay dates, quantity, dollars, freight, approval |
| `purchases` | 2,372,474 | Line-item purchases per PO: brand, quantity, dollars, receiving date |
| `purchase_prices` | 12,261 | Brand-level purchase and retail prices, size, classification |
| `begin_inventory` | 206,529 | Store-level inventory snapshot at start of period |
| `end_inventory` | 224,489 | Store-level inventory snapshot at end of period |

> The `.db` file and trained `.pkl` models are excluded from version control via `.gitignore` because of their size. Place `inventory.db` in `data/` before running.

---

## 🔎 Exploratory Data Analysis

Performed in `notebooks/`.

**Freight cost**
- Freight is very strongly correlated with invoice value: **Dollars ↔ Freight r = 0.985**, Quantity ↔ Freight r = 0.947.
- Freight per unit falls as order size grows (≈ **$0.095/unit** for the smallest 25% of orders vs ≈ **$0.049/unit** for the largest 25%), indicating economies of scale.
- Invoice Dollars was therefore chosen as the single predictive feature.

**Invoice risk flagging**
- Features were engineered by joining each invoice to PO-level aggregates from `purchases`: `total_brands`, `total_item_quantity`, `total_item_dollars`, `avg_receiving_delay`, plus `days_po_to_invoice` and `days_to_pay`.
- Welch's t-tests (flagged vs. normal invoices, α = 0.05) showed `invoice_quantity`, `invoice_dollars`, `Freight`, `days_po_to_invoice`, `total_item_quantity`, `total_item_dollars` and `avg_receiving_delay` differ significantly; `days_to_pay` and `total_brands` did not.
- Class balance: **3,693 normal (66.6%)** vs **1,850 flagged (33.4%)**.

---

## 🤖 Models & Results

All models use an 80/20 train–test split with `random_state=42`.

### 1. Freight Cost Prediction — regression
Feature: `Dollars` → Target: `Freight`

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Linear Regression** ✅ | **24.11** | **124.72** | **96.99%** |
| Random Forest (max_depth=6) | 26.13 | 134.79 | 96.48% |
| Decision Tree (max_depth=5) | 32.97 | 150.31 | 95.63% |

`train.py` selects the model with the lowest MAE automatically, so **Linear Regression** is the saved model.

### 2. Invoice Risk Flagging — classification
**Label rule (rule-based target):** an invoice is flagged (`1`) if
- `|invoice_dollars − total_item_dollars| > $5` (invoice total mismatches line items), **or**
- `avg_receiving_delay > 10` days.

**Features:** `invoice_quantity`, `invoice_dollars`, `Freight`, `total_item_quantity`, `total_item_dollars` (standardised with `StandardScaler`).

| Model (7-feature baseline) | Accuracy | Flagged-class Recall | Flagged-class F1 |
|---|---:|---:|---:|
| Logistic Regression | 65.83% | 0.06 | 0.11 |
| Decision Tree | 81.15% | 0.71 | 0.72 |
| Random Forest | 87.65% | 0.68 | 0.79 |

The final **Random Forest** on the reduced 5-feature set reached **88.46% accuracy** (precision 0.94 / recall 0.72 / F1 0.81 for the flagged class) before tuning. It is then tuned with `GridSearchCV` (5-fold CV, F1 scoring) over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` and `criterion`. The saved model uses `n_estimators=100`, `min_samples_split=5`, `criterion='gini'`, `max_depth=None`.

---

## 🖥 Streamlit Application

`app.py` provides the **Vendor Invoice Intelligence Portal** with a sidebar module selector:

- **Freight Cost Prediction** – enter Quantity and Invoice Dollars, get an estimated freight cost.
- **Invoice Manual Approval Flag** – enter invoice quantity, dollars, freight and PO totals, get a verdict: *Manual Approval required* or *Safe for Auto-Approval*.

---

## 🧱 Project Structure

```
Invoice Intelligence ML Project/
│
├── app.py                          # Streamlit web application
├── requirements.txt
├── .gitignore
│
├── data/
│   └── inventory.db                # SQLite database (not tracked in git)
│
├── notebooks/
│   ├── Predicting Freight Cost.ipynb   # EDA + freight model experiments
│   └── Invoice Flagging.ipynb          # EDA, hypothesis tests, classifier tuning
│
├── freight_cost_prediction/
│   ├── data_preprocessing.py       # Load data, select features, split
│   ├── modeling_evaluation.py      # LR / DT / RF trainers + metrics
│   └── train.py                    # Train, compare, save best model
│
├── invoice_flagging/
│   ├── data_preprocessing.py       # SQL feature build, labelling, split, scaling
│   ├── modeling_evaluation.py      # Random Forest + GridSearchCV + evaluation
│   └── train.py                    # End-to-end training pipeline
│
├── inference/
│   ├── predict_freight.py          # Load model & predict freight
│   └── predict_invoice_flag.py     # Load model & predict risk flag
│
└── models/
    ├── predict_freight_model.pkl   # Linear Regression
    ├── predict_flag_invoice.pkl    # Random Forest Classifier
    └── scaler.pkl                  # StandardScaler for flag features
```

---

## 🚀 How to Run

**1. Clone and set up the environment**
```bash
git clone <your-repo-url>
cd "Invoice Intelligence ML Project"
python -m venv venv
venv\Scripts\activate          # Windows  (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

**2. Add the data** – place `inventory.db` inside `data/`.

**3. Train the models** (run each from inside its own folder)
```bash
cd freight_cost_prediction && python train.py
cd ../invoice_flagging && python train.py
```
> Update the hard-coded database path at the top of each `train.py` / `data_preprocessing.py` to point to your local `data/inventory.db`.

**4. Launch the app**
```bash
streamlit run app.py
```

---

## ⚠️ Known Limitations & Future Work

- **Rule-derived labels.** `flag_invoice` is generated from `invoice_dollars` vs `total_item_dollars` and `avg_receiving_delay`. Since the dollar columns are model inputs, the classifier partly re-learns the labelling rule rather than discovering independent fraud patterns. Using real reviewer decisions (e.g. the `Approval` column) or adding `avg_receiving_delay` / date features would make the model more meaningful.
- **Recall on flagged invoices (~0.72)** leaves room for improvement; consider `class_weight='balanced'`, threshold tuning, or gradient boosting.
- **Single-feature freight model.** Adding vendor, quantity and date features could improve accuracy on large invoices.
- **Packaging.** Replace hard-coded absolute paths with relative/config-based paths, apply the saved scaler inside the inference pipeline (or use a scikit-learn `Pipeline`), and consider Docker + cloud deployment.
- **Monitoring.** Add model drift tracking and a batch-upload (CSV) mode to the app.

---

## 👤 Author

**Ajeyata Maurya**