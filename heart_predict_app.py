from operator import imod
import streamlit as st
import numpy as np
import pandas as pd 
import joblib
import os

model = joblib.load(r'C:\Users\karan\OneDrive\Desktop\daily_practice\heart-disease-app\model\svm_model.pkl')
scaler = joblib.load(r'C:\Users\karan\OneDrive\Desktop\daily_practice\heart-disease-app\model\scaler.pkl')

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("Heart Disease Prediction App")
st.write("Enter patient details below:")

age = st.number_input("Age", min_value=1, max_value=100)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting BP", min_value=80, max_value=200)
chol = st.number_input("Cholesterol", min_value=100, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
restecg = st.selectbox("Resting ECG Result", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250)
exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.number_input("ST depression", min_value=0.0, max_value=10.0)
slope = st.selectbox("Slope of ST", [0, 1, 2])
ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2])


sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High risk of heart disease. Probability: {prob:.2f}")
    else:
        st.success(f"✅ Low risk of heart disease. Probability: {prob:.2f}")
