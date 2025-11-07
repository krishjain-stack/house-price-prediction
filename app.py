# app.py
# ----------------------------
# House Price Prediction using Trained Model
# ----------------------------

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ----------------------------
# Load Model and Scaler
# ----------------------------
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# ----------------------------
# Streamlit Interface
# ----------------------------
st.title("🏡 House Price Prediction App")
st.write("Predict house prices using a trained Linear Regression model.")

st.sidebar.header("Enter House Details")

# Input fields
area = st.sidebar.number_input("Area (sq ft)", min_value=500, max_value=10000, step=100)
bedrooms = st.sidebar.number_input("Bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=1, max_value=10, step=1)
stories = st.sidebar.number_input("Stories", min_value=1, max_value=5, step=1)
parking = st.sidebar.number_input("Parking Spaces", min_value=0, max_value=5, step=1)

# Button to predict
if st.sidebar.button("Predict Price"):
    # Prepare input data
    input_data = np.array([[area, bedrooms, bathrooms, stories, parking]])

    # Scale input
    scaled_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_data)[0]

    st.subheader("🏠 Predicted House Price")
    st.success(f"${prediction:,.2f}")

# ----------------------------
# Optional: Display Sample Dataset
# ----------------------------
if st.checkbox("Show Training Data (sample)"):
    df = pd.read_csv("house_prices.csv")
    st.dataframe(df.head())
