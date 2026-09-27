# Customer Churn Prediction

End-to-end machine learning pipeline that predicts whether a telecom customer
will churn (cancel their subscription), built with scikit-learn.

## What it does
- Cleans and preprocesses customer account data (tenure, contract type,
  monthly charges, tech support usage, payment method, etc.)
- Trains a `RandomForestClassifier`, tuned via `GridSearchCV`
- Benchmarks against a Logistic Regression baseline
- Evaluates with classification report, confusion matrix, and ROC-AUC
- Saves the trained model (`churn_model.joblib`) for reuse/deployment

## Results
- ROC-AUC: ~0.71–0.74 (Random Forest vs. Logistic Regression baseline)
- See `roc_curve.png` and `feature_importance.png` for visuals

## Tech stack
Python, pandas, NumPy, scikit-learn, matplotlib

## How to run
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python customer_churn_prediction.py
```

## Dataset
This repo ships with a synthetic dataset generator so it runs with zero
setup. For a production-grade version, swap in the public
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
dataset from Kaggle and load it with `pd.read_csv()`.

## Future improvements
- Swap in the real Telco dataset
- Add SHAP values for model explainability
- Deploy as a Streamlit app for interactive predictions
