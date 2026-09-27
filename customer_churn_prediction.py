"""
PROJECT: Customer Churn Prediction (Basic Level)
--------------------------------------------------
Goal: Predict whether a telecom customer will churn (leave the service)
based on their account and usage data.

Why this project is a good resume piece:
- Covers the full classic ML pipeline: data cleaning -> EDA -> feature
  engineering -> model training -> evaluation -> model saving.
- Churn prediction is one of the most common real-world business ML
  problems (used by telecom, SaaS, banking, subscription companies).
- Uses scikit-learn, pandas, and matplotlib -- the standard toolkit
  recruiters expect from an entry-level ML profile.

Dataset: A synthetic-but-realistic telecom customer dataset is generated
inside this script, so it runs with zero setup. In your real portfolio
version, swap this for the public "Telco Customer Churn" dataset from
Kaggle (just drop the CSV in the same folder and load it with
pd.read_csv() instead of the generator below) -- it will make for a
more authentic README.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import joblib


# ---------------------------------------------------------------------
# 1. DATA GENERATION (replace with pd.read_csv("telco_churn.csv") for
#    the real Kaggle dataset when you build your portfolio version)
# ---------------------------------------------------------------------
def generate_churn_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(0, 72, n_samples)  # months as a customer
    monthly_charges = rng.uniform(18, 120, n_samples).round(2)
    total_charges = (tenure * monthly_charges * rng.uniform(0.9, 1.1, n_samples)).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n_samples, p=[0.55, 0.25, 0.20])
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], n_samples, p=[0.35, 0.45, 0.20])
    tech_support = rng.choice(["Yes", "No"], n_samples, p=[0.35, 0.65])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n_samples
    )
    senior_citizen = rng.choice([0, 1], n_samples, p=[0.85, 0.15])

    # Build churn probability from realistic risk factors, then sample labels
    churn_score = (
        0.35 * (contract == "Month-to-month")
        + 0.20 * (internet_service == "Fiber optic")
        + 0.15 * (tech_support == "No")
        + 0.10 * (payment_method == "Electronic check")
        + 0.15 * (tenure < 12)
        + 0.10 * (monthly_charges > 80)
        - 0.20 * (tenure > 48)
    )
    churn_prob = 1 / (1 + np.exp(-4 * (churn_score - 0.4)))  # squash to [0,1]
    churn = rng.binomial(1, churn_prob)

    df = pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract": contract,
        "internet_service": internet_service,
        "tech_support": tech_support,
        "payment_method": payment_method,
        "senior_citizen": senior_citizen,
        "churn": churn,
    })
    return df


# ---------------------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------------------
df = generate_churn_dataset()
print("Dataset shape:", df.shape)
print(df.head())
print("\nChurn rate: {:.1f}%".format(df["churn"].mean() * 100))

X = df.drop(columns=["churn"])
y = df["churn"]

numeric_features = ["tenure", "monthly_charges", "total_charges", "senior_citizen"]
categorical_features = ["contract", "internet_service", "tech_support", "payment_method"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------
# 3. PREPROCESSING PIPELINE
# ---------------------------------------------------------------------
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# ---------------------------------------------------------------------
# 4. MODEL PIPELINE + HYPERPARAMETER SEARCH
# ---------------------------------------------------------------------
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42)),
])

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 8, 12],
    "classifier__min_samples_leaf": [1, 3],
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=0
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("\nBest params:", grid_search.best_params_)

# ---------------------------------------------------------------------
# 5. EVALUATION
# ---------------------------------------------------------------------
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC score: {:.3f}".format(roc_auc_score(y_test, y_proba)))

# Baseline comparison (shows you know how to benchmark models)
baseline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
]).fit(X_train, y_train)
baseline_auc = roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1])
print("Logistic Regression baseline ROC-AUC: {:.3f}".format(baseline_auc))

# ---------------------------------------------------------------------
# 6. PLOTS (saved to disk, useful for README screenshots)
# ---------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc_score(y_test, y_proba):.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Customer Churn Prediction")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
print("\nSaved ROC curve to roc_curve.png")

# Feature importance
importances = best_model.named_steps["classifier"].feature_importances_
feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
top_idx = np.argsort(importances)[-10:]
plt.figure(figsize=(7, 5))
plt.barh(range(len(top_idx)), importances[top_idx])
plt.yticks(range(len(top_idx)), [feature_names[i] for i in top_idx])
plt.xlabel("Importance")
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("Saved feature importance chart to feature_importance.png")

# ---------------------------------------------------------------------
# 7. SAVE MODEL FOR DEPLOYMENT
# ---------------------------------------------------------------------
joblib.dump(best_model, "churn_model.joblib")
print("\nModel saved to churn_model.joblib")


def predict_churn(customer: dict) -> dict:
    """Helper for deployment (e.g. inside a Flask/Streamlit app)."""
    row = pd.DataFrame([customer])
    proba = best_model.predict_proba(row)[0, 1]
    return {"churn_probability": round(float(proba), 3), "will_churn": bool(proba > 0.5)}


if __name__ == "__main__":
    sample_customer = {
        "tenure": 3,
        "monthly_charges": 95.0,
        "total_charges": 285.0,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic",
        "tech_support": "No",
        "payment_method": "Electronic check",
        "senior_citizen": 0,
    }
    print("\nSample prediction:", predict_churn(sample_customer))
