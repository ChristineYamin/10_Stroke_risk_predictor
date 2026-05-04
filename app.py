import streamlit as st
import joblib
import pandas as pd
import os

# Set page configuration
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load the model securely using a relative path
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'model', 'stroke_model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Ensure the model folder is uploaded.")
    
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Loading error: {e}")

# Application Title and Description
st.title("🩺 Stroke Risk Prediction System")
st.markdown("""
    This application predicts the likelihood of a stroke using a machine learning model optimized for high recall. 
    Enter the patient details below to get the risk assessment.
""")
st.write("---")

# Layout: Organize inputs into columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics & Lifestyle")
    age = st.number_input("Age", min_value=1, max_value=100, value=50, step=1)
    avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, value=100.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    smoking_status = st.selectbox(
        "Smoking Status",
        ["Never smoked", "Formerly smoked", "Smokes"]
    )

with col2:
    st.subheader("Medical History")
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    ever_married = st.selectbox("Ever Married", ["No", "Yes"])
    work_type = st.selectbox(
        "Work Type",
        ["Private", "Self-employed", "children", "Never_worked", "Govt_job"]
    )
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])

# Process Inputs for the Model
st.write("---")

if st.button("Predict Risk", use_container_width=True):
    # Convert categorical inputs into dummy variables matching training columns
    input_data = {
        'age': [age],
        'hypertension': [1 if hypertension == "Yes" else 0],
        'heart_disease': [1 if heart_disease == "Yes" else 0],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        
        # Gender encoding
        'gender_Male': [1 if gender == "Male" else 0],
        'gender_Other': [1 if gender == "Other" else 0],
        
        # Marital status
        'ever_married_Yes': [1 if ever_married == "Yes" else 0],
        
        # Work type encoding
        'work_type_Never_worked': [1 if work_type == "Never_worked" else 0],
        'work_type_Private': [1 if work_type == "Private" else 0],
        'work_type_Self-employed': [1 if work_type == "Self-employed" else 0],
        'work_type_children': [1 if work_type == "children" else 0],
        
        # Residence type encoding
        'Residence_type_Urban': [1 if residence_type == "Urban" else 0],
        
        # Smoking status encoding
        'smoking_status_formerly smoked': [1 if smoking_status == "Formerly smoked" else 0],
        'smoking_status_never smoked': [1 if smoking_status == "Never smoked" else 0],
        'smoking_status_smokes': [1 if smoking_status == "Smokes" else 0],
    }

    input_df = pd.DataFrame(input_data)
    
    # Predict using the custom threshold established in validation
    prob = model.predict_proba(input_df)[:, 1][0]
    prediction = 1 if prob >= 0.18 else 0

    # Display results
    st.subheader("Prediction Results")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(label="Calculated Risk Probability", value=f"{prob:.1%}")
    
    with col_b:
        if prediction == 1:
            st.error("🚨 HIGH RISK: Patient requires immediate medical evaluation.")
        else:
            st.success("🟢 LOW RISK: No immediate warning signs detected.")
            
    st.caption("Note: This tool is designed to assist medical screening and should not replace professional diagnosis.")