# Customer Churn Prediction

An end-to-end Machine Learning project that predicts whether a customer is likely to churn and explains the factors influencing the prediction.

## Overview

This project implements a complete customer churn prediction pipeline:

- Data preprocessing and cleaning
- Categorical feature encoding and numerical feature scaling
- Training multiple classification models
- Random Forest hyperparameter tuning
- Model evaluation and comparison
- Churn probability and risk-level prediction
- SHAP-based explainability for individual predictions

## Machine Learning Models

The project evaluates:

- Logistic Regression
- Random Forest
- XGBoost

The **tuned Random Forest** was selected as the final model based on its overall performance.

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 76.22% | 53.98% | 70.58% | 61.18% | 84.17% |
| Tuned Random Forest | 76.36% | 53.80% | 77.54% | 63.52% | 84.44% |
| XGBoost | 74.80% | 51.61% | 81.28% | 63.13% | 84.40% |

> The tuned Random Forest achieved an ROC-AUC of 84.44% and was selected as the final model.

## Prediction

The prediction pipeline provides:

- Churn prediction
- Churn probability
- Risk level: Low / Medium / High
- Top factors influencing the prediction

Example:

```text
Prediction        : Likely to Churn
Churn Probability : 77.85%
Risk Level        : High

Top Factors:
- InternetService : Fiber optic → increases churn risk
- Contract : Month-to-month → increases churn risk
- PaymentMethod : Electronic check → increases churn risk
- tenure : 12 → increases churn risk
- OnlineSecurity : No → increases churn risk
```

## Explainability

SHAP is used to understand how features influence the model's predictions.

For an individual customer, SHAP identifies the features that increase or decrease the predicted churn risk.

This makes the model more interpretable instead of providing only a churn prediction.

## Project Structure

```text
Customer-Churn-Prediction/
│
├── data/
│   ├── Telco-Customer-Churn.csv
│   └── processed/
│       ├── x_train.csv
│       ├── x_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_model_training.ipynb
│   ├── churn_preprocessor.joblib
│   └── churn_tuned_rf_model.joblib
│
├── src/
│   └── predict.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Tech Stack

- Python
- Pandas 
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Seaborn
- Jupyter Notebook

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Ashutosh-io7/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebooks

Open Jupyter Notebook or JupyterLab and run:

```text
01_data_prep.ipynb
02_model_training.ipynb
```

### 4. Run a prediction

```bash
python src/predict.py
```

## Key Insights

The model identifies factors such as:

- Tenure
- Contract type
- Internet service
- Payment method
- Online security
- Monthly charges

as important factors in churn predictions.

> SHAP explanations describe how features influence the model's prediction. They should not be interpreted as evidence that a feature directly causes customer churn.