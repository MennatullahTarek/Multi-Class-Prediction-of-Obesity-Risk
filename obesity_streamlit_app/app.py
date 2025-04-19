# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
import pickle

import os
print("available:", os.listdir())

st.set_page_config(page_title="Obesity Risk Classifier", layout="centered")
st.title(" Obesity Risk Classifier ")

# 🧠 Load model and tools
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

model = load_pickle('model.pkl')
scaler = load_pickle('scaler.pkl')
encoder_CAEC = load_pickle('encoder_CAEC.pkl')
encoder_MTRANS = load_pickle('encoder_MTRANS.pkl')
encoder_history = load_pickle('encoder_history.pkl')
label_gender = load_pickle('label_gender.pkl')
ordinal_CALC = load_pickle('ordinal_CALC.pkl')
label_FAVC = load_pickle('label_FAVC.pkl')
label_SCC = load_pickle('label_SCC.pkl')
label_smoke = load_pickle('label_smoke.pkl')
encoder_target = load_pickle('target_encoder.pkl')

# 🎉 Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3875/3875029.png", width=100)
    st.markdown("## 🤖 About the App")
    st.markdown("""
    This app uses **Machine Learning** to predict your **Obesity Risk Level** based on your lifestyle and habits.
    
    Fill out the form and hit **Predict** to see your result!
    """)

# 🏷️ Title
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🍕 Obesity Risk Predictor 🍎</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: gray;'>A smarter way to understand your health.</h5>", unsafe_allow_html=True)
st.write("---")

# 📝 Input Form
with st.form("user_form"):
    st.subheader("👤 Personal Information")
    col1, col2 = st.columns(2)

    with col1:
        Gender = st.selectbox("👫 Gender", ['Male', 'Female'])
        Age = st.slider("🎂 Age", 10, 100, 25)
        Height = st.number_input("📏 Height (m)", min_value=1.0, max_value=2.5, value=1.65)
        Weight = st.number_input("⚖️ Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)

    with col2:
        family_history_with_overweight = st.selectbox("👨‍👩‍👧 Family History with Overweight", ['Yes', 'No'])
        FAVC = st.selectbox("🍔 Frequent High-Calorie Food (FAVC)", ['Yes', 'No'])
        FCVC = st.slider("🥦 Vegetable Consumption Frequency (1-3)", 1.0, 3.0, 2.0)
        NCP = st.slider("🍽️ Meals per Day", 1.0, 5.0, 3.0)

    st.subheader("🏃 Lifestyle & Habits")
    col3, col4 = st.columns(2)

    with col3:
        CAEC = st.selectbox("🍫 Snacking Frequency (CAEC)", ['Never', 'Sometimes', 'Frequently', 'Always'])
        SMOKE = st.selectbox("🚬 Do you smoke?", ['Yes', 'No'])
        CH2O = st.slider("💧 Water Intake (liters)", 0.0, 5.0, 2.0)

    with col4:
        SCC = st.selectbox("📉 Calorie Monitoring (SCC)", ['Yes', 'No'])
        FAF = st.slider("🏋️‍♂️ Physical Activity (0-3)", 0.0, 3.0, 1.0)
        TUE = st.slider("📱 Time on Devices (0-2)", 0.0, 2.0, 1.0)

    CALC = st.selectbox("🍷 Alcohol Consumption", ['Never', 'Sometimes', 'Frequently', 'Always'])
    MTRANS = st.selectbox("🚌 Main Transport Mode", ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])

    submit = st.form_submit_button("🔍 Predict My Risk!")

if submit:
    input_dict = {
        'Gender': [Gender],
        'Age': [Age],
        'Height': [Height],
        'Weight': [Weight],
        'family_history_with_overweight': [family_history_with_overweight],
        'FAVC': [FAVC],
        'FCVC': [FCVC],
        'NCP': [round(NCP)],
        'CAEC': [CAEC],
        'SMOKE': [SMOKE],
        'CH2O': [CH2O],
        'SCC': [SCC],
        'FAF': [FAF],
        'TUE': [TUE],
        'CALC': [CALC],
        'MTRANS': [MTRANS],
    }

    df = pd.DataFrame(input_dict)

    # 🔧 Preprocessing
    col_numerical = ['Age', 'Height', 'Weight', 'FCVC', 'CH2O', 'FAF', 'TUE']
    df[col_numerical] = scaler.transform(df[col_numerical])
    df['CAEC'] = encoder_CAEC.transform(df[['CAEC']])
    df['MTRANS'] = encoder_MTRANS.transform(df[['MTRANS']])
    df['family_history_with_overweight'] = encoder_history.transform(df[['family_history_with_overweight']])
    df['Gender'] = label_gender.transform(df['Gender'])
    df['CALC'] = ordinal_CALC.transform(df[['CALC']])
    df['FAVC'] = label_FAVC.transform(df['FAVC'])
    df['SCC'] = label_SCC.transform(df['SCC'])
    df['SMOKE'] = label_smoke.transform(df['SMOKE'])

    # 🧠 Prediction
    prediction = model.predict(df)
    label = encoder_target.inverse_transform(prediction)

    # 🎯 Result
    st.success(f"🎉 Your Predicted Obesity Risk Level is: **{label[0]}**")
    st.balloons()
    st.markdown("Stay healthy and take care of yourself! 💚")

