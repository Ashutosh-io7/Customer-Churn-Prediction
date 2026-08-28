import os 
import joblib 
import pandas as pd 
import shap 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

preprocessor = joblib.load(
    os.path.join(BASE_DIR , "notebooks" , "churn_preprocessor.joblib")
) 

model = joblib.load(
    os.path.join(BASE_DIR , "notebooks" , "churn_tuned_rf_model.joblib")
) 

explainer = shap.TreeExplainer(model) 

feature_names = preprocessor.get_feature_names_out()

def predict_churn(customer_data) : 

    required_features = preprocessor.feature_names_in_ 

    missing_features = [
        feature for feature in required_features 
        if feature not in customer_data 
    ] 

    if missing_features : 
        raise ValueError(
            f"Missing Required Features : {missing_features}"
        )


    data = pd.DataFrame([customer_data]) 

    transformed_data = preprocessor.transform(data)

    probability = model.predict_proba(transformed_data)[0][1] 

    shap_values = explainer.shap_values(transformed_data) 

    if len(shap_values.shape) == 3 : 
        customer_shap = shap_values[0, :, 1]
    else : 
        customer_shap = shap_values[0] 

    feature_groups = {} 

    for transformed_feature in feature_names : 
        clean_name = transformed_feature.split("__", 1)[-1]

        matching_columns = [
            column for column in data.columns 
            if clean_name == column or clean_name.startswith(column + "_")
        ] 

        if matching_columns : 
            original_feature = max(matching_columns, key = len) 
            feature_groups.setdefault(original_feature, []).append(
                transformed_feature
            )
    feature_impacts = {} 

    for original_feature, transformed_features in feature_groups.items():
        indices = [
            list(feature_names).index(feature)
            for feature in transformed_features
        ]

        feature_impacts[original_feature] = customer_shap[indices].sum()

    # Get top 5 original features
    top_features = sorted(
        feature_impacts.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    prediction = int(probability >= 0.5) 

    if probability < 0.30 : 
        risk_level = "Low" 
    elif probability < 0.70 : 
        risk_level = "Medium" 
    else : 
        risk_level = "High"

    return {
        "prediction" : "Likely to Churn" if prediction == 1 else "Likely to Stay",
        "churn_probability" : round(probability , 4),
        "risk_level" : risk_level,
        "top_factors" : [
            {
                "feature" : feature,
                "value" : data.iloc[0][feature],
                "impact" : round(float(impact), 4),
                "direction" : "increases" if impact > 0 else "decreases"
            }
            for feature, impact in top_features 
        ] 
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
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-Month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.20
    }

    result = predict_churn(customer) 

    print("Prediction :" , result['prediction']) 
    print("Churn Probability :" , f"{result['churn_probability']:.2%}")
    print("Risk Level :" , result['risk_level']) 

    print("\n Top Factors :") 

    for factor in result['top_factors'] : 
        print(
            f"- {factor['feature']} : {factor['value']}" 
            f" -> {factor['direction']} Churn Risk" 
        )