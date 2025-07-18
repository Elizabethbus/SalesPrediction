import streamlit as st
import joblib
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# App layout
st.set_page_config(page_title="🇳🇬 Nigerian SME Sales Predictor", layout="centered")
st.title(" Predict Daily Sales for Nigerian SMEs")
st.markdown("""
This tool helps Nigerian small businesses forecast daily sales using a machine learning model trained on real e-commerce data.
""")

# Form for user input
with st.form("sales_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        day = st.number_input("Day of Month", min_value=1, max_value=31, value=15)
        month = st.number_input("Month", min_value=1, max_value=12, value=7)
    
    with col2:
        is_weekend = st.selectbox("Is it a Weekend?", ["No", "Yes"])
        promo_day = st.selectbox("Is it a Promo Day?", ["No", "Yes"])
    
    weather = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Rainy"])
    
    submitted = st.form_submit_button("Predict")

# Predict
if submitted:
    weekend = 1 if is_weekend == "Yes" else 0
    promo = 1 if promo_day == "Yes" else 0
    weather_val = {"Sunny": 0, "Cloudy": 1, "Rainy": 2}[weather]
    
    features = np.array([[day, month, weekend, promo, weather_val]])
    prediction = model.predict(features)[0]
    
    st.success(f"📈 Predicted Sales: ₦{prediction:,.2f}")
