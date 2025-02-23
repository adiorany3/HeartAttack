import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import datetime

# Load the trained model and scaler
model_path = 'model.pkl'
scaler_path = 'scaler.pkl'
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Function to make predictions
def predict_heart_attack(input_data):
    input_data_scaled = scaler.transform([input_data])
    prediction = model.predict(input_data_scaled)
    return prediction[0]

# Streamlit app layout
st.title('Heart Attack Prediction App')
st.markdown('This prediction system uses the Kaggle dataset with the link [here](https://www.kaggle.com/datasets/ankushpanday2/heart-attack-prediction-in-united-states?resource=download)')

# User input for features
age = st.number_input('Age', min_value=0, max_value=120)
gender = st.selectbox('Gender', options=['Male', 'Female'])
cholesterol = st.number_input('Cholesterol Level', min_value=0)
blood_pressure = st.number_input('Blood Pressure', min_value=0)
heart_rate = st.number_input('Heart Rate', min_value=0)
smoker = st.selectbox('Smoker', options=['Yes', 'No'])
diabetes = st.selectbox('Diabetes', options=['Yes', 'No'])
hypertension = st.selectbox('Hypertension', options=['Yes', 'No'])
family_history = st.selectbox('Family History', options=['Yes', 'No'])
stress_level = st.number_input('Stress Level (0=low ; 9=Very Stress)', min_value=0)

# Convert categorical inputs to numerical
gender = 1 if gender == 'Male' else 0
smoker = 1 if smoker == 'Yes' else 0
diabetes = 1 if diabetes == 'Yes' else 0
hypertension = 1 if hypertension == 'Yes' else 0
family_history = 1 if family_history == 'Yes' else 0

# Create input data array
input_data = [age, gender, cholesterol, blood_pressure, heart_rate, smoker, diabetes, hypertension, family_history, stress_level]

# Button to make prediction
if st.button('Predict'):
    result = predict_heart_attack(input_data)
    if result == 1:
        st.success('The model predicts a high risk of heart attack.')
        st.write("Please consult a doctor immediately.") # Additional message
    else:
        st.success('The model predicts a low risk of heart attack.')
        st.write("Maintain a healthy lifestyle.") # Additional message

# Get the current year
current_year = datetime.datetime.now().year

# Footer
st.markdown(f"""
<div style="text-align: center; padding-top: 20px;">
    © {current_year} Developed by: Galuh Adi Insani. All rights reserved.
</div>
""", unsafe_allow_html=True)