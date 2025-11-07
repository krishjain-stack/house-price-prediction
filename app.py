import streamlit as st
import pickle
import numpy as np

# Load your model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🏠 House Price Prediction App")

st.write("Enter the input features below:")

# Example input fields — replace or add more according to your model
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)
# Add as many as needed based on your model’s training features

if st.button("Predict"):
    try:
        features = np.array([[feature1, feature2, feature3]])  # match model input size
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)
        st.success(f"Predicted Price: ₹{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")

