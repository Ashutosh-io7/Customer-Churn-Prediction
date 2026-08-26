import os 
import joblib 
import pandas as pd 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

preprocessor = joblib.load(
    os.path.join(BASE_DIR , "notebooks" , "churn_preprocessor.joblib")
) 

model = joblib.load(
    os.path.join(BASE_DIR , "notebooks" , "churn_tuned_rf_model.joblib")
) 

def predict_churn(customer_data) : 
    data = pd.DataFrame([customer_data]) 

    transformed_data = preprocessor.transform(data)

    probability = model.predict_proba(transformed_data)[0][1] 

    prediction = int(probability >= 0.5) 

    return {
        "prediction" : "Likely to Churn" if prediction == 1 else "Likely to Stay",
        "churn_probability" : round(probability , 4) 
    } 

if __name__ == "__main__" : 

    customer = {
        "gender" : "Male",
        "SeniorCitizen" : 0,
        "Partner" : "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.20
    }

    result = predict_churn(customer) 

    print("Prediction :" , result['prediction']) 
    print("Churn Probability :" , result['churn_probability'])