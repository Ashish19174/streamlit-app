import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import os

# DEBUG – Show where Streamlit is running
st.write("DEBUG — Current Working Directory:", os.getcwd())
st.write("DEBUG — Files in this folder:", os.listdir())

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Predictor (California Housing Model)")
st.markdown("""
This Streamlit app loads a pre-trained house price model (`best_house_price_model.pkl`)
and a scaler (`scaler.pkl`) to predict median house value for a block group.
**Make sure `best_house_price_model.pkl`, `scaler.pkl`, and `feature_names.pkl` are in the same folder as this app.**
""")

# Load model and scaler
MODEL_FILE = "best_house_price_model.pkl"
SCALER_FILE = "scaler.pkl"
FEATURES_FILE = "feature_names.pkl"

@st.cache_resource
def load_artifacts():
    artifacts = {}
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            artifacts["model"] = pickle.load(f)
    else:
        artifacts["model"] = None
    if os.path.exists(SCALER_FILE):
        with open(SCALER_FILE, "rb") as f:
            artifacts["scaler"] = pickle.load(f)
    else:
        artifacts["scaler"] = None
    if os.path.exists(FEATURES_FILE):
        with open(FEATURES_FILE, "rb") as f:
            artifacts["feature_names"] = pickle.load(f)
    else:
        artifacts["feature_names"] = None
    return artifacts

art = load_artifacts()
model = art.get("model")
scaler = art.get("scaler")
feature_names = art.get("feature_names")

if model is None:
    st.warning("Model file `best_house_price_model.pkl` not found. Place it next to this app and rerun.")
if scaler is None:
    st.warning("Scaler file `scaler.pkl` not found. Place it next to this app and rerun.")
if feature_names is None:
    st.info("Feature names file `feature_names.pkl` not found. Predictions will still attempt if model and scaler exist.")

st.sidebar.header("Input Features")
med_inc = st.sidebar.slider("Median Income (MedInc)", 0.5, 15.0, 3.5, 0.1)
house_age = st.sidebar.slider("House Age", 1, 100, 30)
ave_rooms = st.sidebar.slider("Average Rooms", 0.5, 50.0, 5.5, 0.1)
ave_bedrms = st.sidebar.slider("Average Bedrooms", 0.1, 30.0, 1.0, 0.1)
population = st.sidebar.number_input("Population", min_value=1, max_value=100000, value=1200)
ave_occup = st.sidebar.slider("Average Occupants", 1.0, 20.0, 3.0, 0.1)
latitude = st.sidebar.slider("Latitude", -90.0, 90.0, 34.0, 0.01)
longitude = st.sidebar.slider("Longitude", -180.0, 180.0, -118.0, 0.01)

if st.button("Predict"):
    if model is None or scaler is None:
        st.error("Model or scaler not loaded. Ensure the files exist in the app folder.")
    else:
        # Prepare input
        input_df = pd.DataFrame([{
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms,
            "Population": population,
            "AveOccup": ave_occup,
            "Latitude": latitude,
            "Longitude": longitude
        }])
        # Engineered features
        input_df["ROOMS_PER_HOUSEHOLD"] = input_df["AveRooms"] / input_df["AveOccup"]
        input_df["BEDROOMS_RATIO"] = input_df["AveBedrms"] / input_df["AveRooms"]
        input_df["POPULATION_PER_HOUSEHOLD"] = input_df["Population"] / input_df["HouseAge"]
        
        # Ensure column order if available
        if feature_names is not None:
            input_df = input_df[feature_names]
        # Scale
        input_scaled = scaler.transform(input_df)
        # Predict
        pred = model.predict(input_scaled)[0] * 100000  # convert to dollars
        st.success(f"🏠 Predicted House Price: ${pred:,.2f}")
        st.write("**Note:** Model was trained on California housing data; results are median block-group values.")

st.markdown("---")
st.markdown("**How to use**: put `best_house_price_model.pkl`, `scaler.pkl`, and `feature_names.pkl` in the same directory as this app, then run:")
st.code("streamlit run streamlit_app.py", language="bash")
st.markdown("You can customize UI or add charts (actual vs predicted, feature importance) as needed.")

