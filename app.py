# app.py
# ----------------------------
# House Price Prediction App
# Using Streamlit + scikit-learn
# ----------------------------

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import math

# ----------------------------
# Load Data
# ----------------------------
st.title("🏡 House Price Prediction App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your house dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # ----------------------------
    # Model Training
    # ----------------------------
    if "Price" not in df.columns:
        st.error("Dataset must contain a column named 'Price'")
    else:
        X = df.drop("Price", axis=1)
        y = df["Price"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Evaluation
        mae = abs(y_test - y_pred).mean()
        rmse = math.sqrt(((y_test - y_pred) ** 2).mean())
        r2 = model.score(X_test, y_test)

        st.subheader("📈 Model Performance")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

        # ----------------------------
        # Prediction Form
        # ----------------------------
        st.subheader("🔮 Predict House Price")

        with st.form("prediction_form"):
            area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, step=100)
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, step=1)
            bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, step=1)
            stories = st.number_input("Stories", min_value=1, max_value=5, step=1)
            parking = st.number_input("Parking Spaces", min_value=0, max_value=5, step=1)

            predict_button = st.form_submit_button("Predict Price")

        if predict_button:
            new_data = [[area, bedrooms, bathrooms, stories, parking]]
            prediction = model.predict(new_data)
            st.success(f"🏠 Estimated House Price: **${prediction[0]:,.2f}**")

else:
    st.info("👆 Please upload your CSV file to start (must contain `Price` column).")
