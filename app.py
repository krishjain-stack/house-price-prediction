# app.py - robust version to handle scaler/model feature mismatches
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import traceback
import json

st.set_page_config(layout="centered")
st.title("🏡 House Price Prediction (robust)")

# --- Load model & scaler ---
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PKL = "feature_columns.pkl"      # optional (pickle list of column names)
FEATURES_JSON = "feature_columns.json"    # optional (json list of column names)

model = None
scaler = None
saved_feature_names = None

# helper to load safely
def try_load_pickle(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def try_load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

# load model and scaler (show helpful errors if missing)
try:
    model = try_load_pickle(MODEL_PATH)
    if model is None:
        st.error(f"Model file not found: {MODEL_PATH}")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

try:
    scaler = try_load_pickle(SCALER_PATH)
    if scaler is None:
        st.warning(f"Scaler file not found: {SCALER_PATH} — continuing without scaling.")
except Exception as e:
    st.warning(f"Error loading scaler: {e} — continuing without scaling.")
    scaler = None

# try load saved feature columns if present (best practice)
saved_feature_names = try_load_pickle(FEATURES_PKL) or try_load_json(FEATURES_JSON)

if saved_feature_names is not None:
    st.info("Loaded saved feature columns for input alignment.")
    st.write(f"Saved feature count: {len(saved_feature_names)}")
    # optional: show first few saved features
    st.write("Sample saved features:", saved_feature_names[:20])

st.sidebar.header("Enter House Details")
area = st.sidebar.number_input("Area (sq ft)", min_value=100, max_value=20000, value=2600, step=50)
bedrooms = st.sidebar.number_input("Bedrooms", min_value=0, max_value=10, value=3, step=1)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=0, max_value=10, value=2, step=1)
stories = st.sidebar.number_input("Stories", min_value=1, max_value=10, value=2, step=1)
parking = st.sidebar.number_input("Parking Spaces", min_value=0, max_value=10, value=1, step=1)

predict = st.sidebar.button("Predict Price")

def build_input_vector_from_saved_features(saved_names, small_input_map):
    """
    Create a full-length input vector matching saved_names.
    small_input_map: dict of values for core features (area, bedrooms, ...)
    For any missing feature in small_input_map, we fill with 0 (or a reasonable default).
    """
    vec = []
    for name in saved_names:
        if name.lower() in ("area", "sqft", "size", "area_sqft"):
            vec.append(small_input_map.get("Area", 0))
        elif name.lower() in ("bedrooms", "beds", "br"):
            vec.append(small_input_map.get("Bedrooms", 0))
        elif name.lower() in ("bathrooms", "baths", "ba"):
            vec.append(small_input_map.get("Bathrooms", 0))
        elif name.lower() in ("stories", "floors"):
            vec.append(small_input_map.get("Stories", 0))
        elif name.lower() in ("parking", "parking_spaces", "garage"):
            vec.append(small_input_map.get("Parking", 0))
        else:
            # Unknown column from training (maybe a one-hot or engineered feature) -> fill with 0
            vec.append(0)
    return np.array(vec).reshape(1, -1)

if predict:
    input_map = {"Area": area, "Bedrooms": bedrooms, "Bathrooms": bathrooms, "Stories": stories, "Parking": parking}
    user_input = np.array([[area, bedrooms, bathrooms, stories, parking]])
    st.write("### Input summary")
    st.write(input_map)
    try:
        # If no scaler saved -> attempt to predict directly (but warn)
        if scaler is None:
            st.warning("No scaler loaded. Attempting to predict without scaling (may be inaccurate).")
            prediction = model.predict(user_input)[0]
            st.success(f"🏠 Predicted Price (no scaling): ${prediction:,.2f}")
        else:
            # check how many features the scaler expects
            n_expected = getattr(scaler, "n_features_in_", None)
            input_shape = user_input.shape[1]

            if n_expected is None:
                # older sklearn versions may not have n_features_in_ attribute -> attempt direct transform
                try:
                    scaled = scaler.transform(user_input)
                    prediction = model.predict(scaled)[0]
                    st.success(f"🏠 Predicted Price: ${prediction:,.2f}")
                except Exception as e:
                    st.error("Scaler transform failed. See debug info below.")
                    st.error(traceback.format_exc())
            else:
                # If scaler expects exactly our 5 features -> OK
                if n_expected == input_shape:
                    scaled = scaler.transform(user_input)
                    prediction = model.predict(scaled)[0]
                    st.success(f"🏠 Predicted Price: ${prediction:,.2f}")
                else:
                    # MISMATCH: try to reconcile using saved_feature_names if available
                    st.error(f"Feature count mismatch: scaler expects {n_expected} features but input has {input_shape}.")
                    if saved_feature_names:
                        st.info("Attempting to build a full feature vector from saved feature names...")
                        full_vec = build_input_vector_from_saved_features(saved_feature_names, input_map)
                        if full_vec.shape[1] == n_expected:
                            st.write("Built vector shape matches scaler expectation. Proceeding to scale & predict.")
                            scaled = scaler.transform(full_vec)
                            prediction = model.predict(scaled)[0]
                            st.success(f"🏠 Predicted Price (via aligned features): ${prediction:,.2f}")
                        else:
                            st.error("Failed to build matching vector. Built vector shape = "
                                     f"{full_vec.shape[1]} (expected {n_expected}).")
                            st.write("Built vector sample (first 20):", full_vec.ravel()[:20].tolist())
                    else:
                        st.error("No saved feature column list (feature_columns.pkl/json) found. "
                                 "Cannot automatically align features.")
                        st.write("You should re-create and save the feature list used during training (recommended).")
                        st.write("Scaler `n_features_in_`:", n_expected)
                        st.write("Your input shape:", user_input.shape)
                        st.write("Options:")
                        st.write("1) Retrain scaler/model to expect the 5 core features and save them.")
                        st.write("2) Save the full feature column list used during training as `feature_columns.pkl` "
                                 "or `feature_columns.json` and push to repo; I will then align automatically.")
    except Exception as e:
        st.error("An unexpected error occurred; full traceback below.")
        st.error(traceback.format_exc())
