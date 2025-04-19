# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Get the current directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Load pickle helper
def load_pickle(filename):
    filepath = os.path.join(BASE_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return joblib.load(f)
    else:
        st.error(f"❌ File not found: {filename}")
        raise FileNotFoundError(f"{filename} not found in {BASE_DIR}")

# 🔄 Load model and tools
model = load_pickle('model.pkl')
scaler = load_pickle('scaler.pkl')  # If you used scaling
encoder_CAEC = load_pickle('encoder_caec.pkl')
encoder_MTRANS = load_pickle('encoder_mtrans.pkl')
encoder_history = load_pickle('encoder_history.pkl')
label_gender = load_pickle('label_gender.pkl')
ordinal_CALC = load_pickle('encoder_calc.pkl')
label_FAVC = load_pickle('label_favc.pkl')
label_SCC = load_pickle('label_scc.pkl')
label_smoke = load_pickle('label_smoke.pkl')
encoder_target = load_pickle('target_encoder.pkl')  # Assuming you encoded your y labels

# 🎨 Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3875/3875029.png", width=100)
    st.markdown("## 🤖 About the App")
    st.markdown("""This app uses **Machine Learning** to predict your **Obesity Risk Level** based on your lifestyle and habits.
    Fill out the form and hit **Predict** to see your result!""")

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
        family_history_with_overweight = st.selectbox("👨‍👩‍👧 Family History with Overweight", ['yes', 'no'])
        FAVC = st.selectbox("🍔 Frequent High-Calorie Food (FAVC)", ['yes', 'no'])
        FCVC = st.slider("🥦 Vegetable Consumption Frequency (1-3)", 1.0, 3.0, 2.0)
        NCP = st.slider("🍽️ Meals per Day", 1.0, 5.0, 3.0)

    st.subheader("🏃 Lifestyle & Habits")
    col3, col4 = st.columns(2)

    with col3:
        CAEC = st.selectbox("🍫 Snacking Frequency (CAEC)", ['no', 'Sometimes', 'Frequently', 'Always'])
        SMOKE = st.selectbox("🚬 Do you smoke?", ['yes', 'no'])
        CH2O = st.slider("💧 Water Intake (liters)", 0.0, 5.0, 2.0)

    with col4:
        SCC = st.selectbox("📉 Calorie Monitoring (SCC)", ['yes', 'no'])
        FAF = st.slider("🏋️‍♂️ Physical Activity (0-3)", 0.0, 3.0, 1.0)
        TUE = st.slider("📱 Time on Devices (0-2)", 0.0, 2.0, 1.0)

    CALC = st.selectbox("🍷 Alcohol Consumption", ['no', 'Sometimes', 'Frequently', 'Always'])
    MTRANS = st.selectbox("🚌 Main Transport Mode", ['Walking', 'Bike', 'Motorbike', 'Public_Transportation', 'Automobile'])

    submit = st.form_submit_button("🔍 Predict My Risk!")

# 🔮 Prediction
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

    # 🔄 Apply transformations
    col_numerical = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    df[col_numerical] = scaler.transform(df[col_numerical])  # Assuming you saved it

    df['CAEC'] = encoder_CAEC.transform(df[['CAEC']])
    df['MTRANS'] = encoder_MTRANS.transform(df[['MTRANS']])
    df['family_history_with_overweight'] = encoder_history.transform(df[['family_history_with_overweight']])
    df['Gender'] = label_gender.transform(df['Gender'])
    df['CALC'] = ordinal_CALC.transform(df[['CALC']])
    df['FAVC'] = label_FAVC.transform(df['FAVC'])
    df['SCC'] = label_SCC.transform(df['SCC'])
    df['SMOKE'] = label_smoke.transform(df['SMOKE'])

    # 🧠 Predict
    prediction = model.predict(df)
    label = encoder_target.inverse_transform(prediction.reshape(-1, 1))

   
    # 🎉 Output
    predicted_label = label[0]
    st.success(f"🎉 Your Predicted Obesity Risk Level is: **{predicted_label}**")
    st.balloons()
    st.markdown("Stay healthy and take care of yourself! 💚")


