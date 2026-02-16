import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
pipeline = joblib.load("churn_pipeline.pkl")

st.title("Customer Churn Predictor")
st.write("Enter customer details to predict churn:")

# --- User Inputs ---
gender = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", [0, 1])
dependents = st.selectbox("Dependents", [0, 1])
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])

# --- Prepare input DataFrame ---
input_data = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract,
    "InternetService": internet,
    "DeviceProtection": device_protection,
    "PaperlessBilling": paperless,
    "PhoneService": phone_service
}])

# --- Ensure correct types (numeric columns as float) ---
numeric_cols = ['SeniorCitizen','Partner','Dependents','tenure','MonthlyCharges','TotalCharges']
input_data[numeric_cols] = input_data[numeric_cols].astype(float)

# --- Predict button ---
if st.button("Predict Churn"):
    try:
        # Use the pipeline directly (preprocessing included)
        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][1]

        st.write("---")
        st.write(f"**Probability of churn:** {probability:.2f}")
        if prediction == 1:
            st.error("⚠️ Customer is likely to churn!")
        else:
            st.success("✅ Customer is likely to stay.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
