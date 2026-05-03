import streamlit as st
import joblib
import pandas as pd

# 1. Load the model
model = joblib.load('C:/Projects/10_Stroke_risk_prediction/model/stroke_model.pkl')

st.title("🧠 Stroke Risk Prediction System")
st.write("This app predicts the likelihood of a stroke based on health metrics.")

# 2. Collect user inputs (Continuous features)
age = st.number_input("Age", min_value=0, max_value=120, value=45)
avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])

# Collect categorical inputs
gender = st.selectbox("Gender", ["Female", "Male", "Other"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
smoking_status = st.selectbox("Smoking Status", ["smokes", "formerly smoked", "never smoked", "Unknown"])

# 3. Reconstruct the 16-feature input DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'gender_Male': [1 if gender == "Male" else 0],
    'gender_Other': [1 if gender == "Other" else 0],
    'ever_married_Yes': [1 if ever_married == "Yes" else 0],
    'work_type_Never_worked': [1 if work_type == "Never_worked" else 0],
    'work_type_Private': [1 if work_type == "Private" else 0],
    'work_type_Self-employed': [1 if work_type == "Self-employed" else 0],
    'work_type_children': [1 if work_type == "children" else 0],
    'Residence_type_Urban': [1 if Residence_type == "Urban" else 0],
    'smoking_status_formerly smoked': [1 if smoking_status == "formerly smoked" else 0],
    'smoking_status_never smoked': [1 if smoking_status == "never smoked" else 0],
    'smoking_status_smokes': [1 if smoking_status == "smokes" else 0]
})

# 4. Predict button
if st.button("Predict Risk"):
    # Get probabilities
    prob = model.predict_proba(input_data)[:, 1]
    
    # Apply our custom threshold (0.18)
    prediction = (prob >= 0.18).astype(int)
    
    if prediction[0] == 1:
        st.error(f"High Risk Detected! Probability: {prob[0]:.2f}")
    else:
        st.success(f"Low Risk. Probability: {prob[0]:.2f}")