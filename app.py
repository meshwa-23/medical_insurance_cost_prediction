import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("best_model_pipeline.joblib")

st.set_page_config(page_title="Medical Insurance Cost Predictor", page_icon="💰", layout="centered")

st.markdown("<h1 style='text-align: center; color: #2E86C1;'>💰 Medical Insurance Cost Predictor</h1>", unsafe_allow_html=True)
st.write("Fill in the details below to estimate your **health insurance cost** (in INR).")

# Sidebar for inputs
st.sidebar.header("User Input Features")

age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.slider("Number of Children", 0, 5, 0)
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Prediction
if st.sidebar.button("💡 Predict"):
    input_df = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    usd_cost = model.predict(input_df)[0]
    inr_cost = usd_cost * 83  # USD → INR

    st.markdown(
        f"<h2 style='text-align: center; color: green;'>Estimated Insurance Cost: ₹{inr_cost:,.2f}</h2>",
        unsafe_allow_html=True
    )
