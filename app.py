# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

st.set_page_config(page_title="ANFDR - Nano Fertilizer Recommender", layout="wide")

st.title("🌱🥬 ANFDR – AI Nano Fertilizer Dosage Regulator 🌿🌾")

# -------------------------
# Local Weather Display
# -------------------------


with st.container():
    st.markdown("""
    <div style='background-color:#8FB3E2;padding:15px;border-radius:12px'>
    <h3>🌦 Local Weather (Live)</h3>
    </div>
    """, unsafe_allow_html=True)

    city = st.text_input("City", "Bangalore")

    try:
        geo = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1",
            timeout=5
        ).json()

        if "results" in geo and len(geo["results"]) > 0:
            lat = geo["results"][0]["latitude"]
            lon = geo["results"][0]["longitude"]

            weather_data = requests.get(
                f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true",
                timeout=5
            ).json()

            current = weather_data.get("current_weather", None)
            if current:
                st.success(
                    f"🌡 {current['temperature']} °C | "
                    f"💨 Wind: {current['windspeed']} km/h | "
                    f"🧭 Direction: {current['winddirection']}°"
                )
            else:
                st.warning("Weather data unavailable")
        else:
            st.warning("City not found")
    except Exception as e:
        st.warning(f"Weather unavailable ({e})")



# -------------------------
# Config & Constants
# -------------------------
MODEL_PATH = "nano_model.joblib"
SAFE_MAX_BY_CROP = {
    "generic": 2.0,
    "fenugreek": 2.0,
    "mint": 1.8,
    "coriander": 2.0,
    "spinach": 1.5,
    "tomato": 2.5
}

CROPS = ["generic","fenugreek","mint","coriander","spinach","tomato"]
SOIL_TYPES = ["red soil", "black soil", "laterite soil", "loamy soil", "clayey soil", "sandy soil"]

# -------------------------
# Synthetic Dataset Generator
# -------------------------
def generate_synthetic_data(n=4000, seed=42):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "crop": rng.choice(CROPS, n),
        "sunlight_hrs": rng.uniform(3, 10, n),
        "temp_c": rng.uniform(15, 38, n),
        "soil_ph": rng.uniform(4.8, 7.8, n),
        "soil_moisture": rng.uniform(15, 60, n),
        "plant_age_days": rng.randint(7, 70, n),
        "npk_amount": rng.uniform(50, 400, n),
        "soil_type": rng.choice(SOIL_TYPES, n),
        "deficiency_score": rng.randint(0, 4, n)
    })

    # Conservative dose logic
    dose = (
    0.25
    + 0.5 * (df["deficiency_score"] / 3)          # deficiency impact
    + 0.25 * (1 - (df["npk_amount"] / 300).clip(0, 1))  # ⬅ STRONG NPK EFFECT
    + 0.08 * (6.5 - df["soil_ph"]).clip(0)
    - 0.003 * df["plant_age_days"]
)

    dose = np.clip(dose, 0.05, 2.0) # safe cap
    df["nano_amount_ml"] = dose.round(3)
    return df

# -------------------------
# Train Model
# -------------------------
def train_model(df, path=MODEL_PATH):
    X = df.drop(columns=["nano_amount_ml"])
    y = df["nano_amount_ml"]

    num_cols = ["sunlight_hrs","temp_c","soil_ph","soil_moisture","plant_age_days","npk_amount","deficiency_score"]
    cat_cols = ["crop","soil_type"]

    preproc = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    pipe = Pipeline([("preproc", preproc), ("model", model)])

    pipe.fit(X, y)
    joblib.dump({"pipeline": pipe, "features": X.columns.tolist()}, path)
    return pipe

# -------------------------
# Load or Train Initial Model
# -------------------------
if os.path.exists(MODEL_PATH):
    bundle = joblib.load(MODEL_PATH)
    model = bundle["pipeline"]
    features = bundle["features"]
else:
    synth_df = generate_synthetic_data()
    model = train_model(synth_df)
    features = synth_df.drop(columns=["nano_amount_ml"]).columns.tolist()
    bundle = joblib.load(MODEL_PATH)

st.success(f"Model trained on synthetic data.")

# -------------------------
# Farmer Prediction Form
# -------------------------
with st.form("predict_form"):
    crop = st.selectbox("Crop", CROPS)
    soil_type = st.selectbox("Soil Type", SOIL_TYPES)

    c1, c2, c3 = st.columns(3)
    sunlight = c1.number_input("Sunlight (hrs/day)", 0.0, 12.0, 6.0)
    temp = c2.number_input("Temperature (°C)", 0.0, 50.0, 28.0)
    ph = c3.number_input("Soil pH", 3.0, 9.0, 6.5)

    c4, c5, c6 = st.columns(3)
    moisture = c4.number_input("Soil Moisture (%)", 0.0, 100.0, 40.0)
    age = c5.number_input("Plant Age (days)", 1, 120, 30)
    npk = c6.number_input("Soil NPK (mg/kg)", 0.0, 1000.0, 180.0)

    deficiency = st.selectbox("Deficiency Score", [0,1,2,3])
    plants = st.number_input("Number of plants", 1, 10000, 10)

    submit = st.form_submit_button("🌱 Predict Dosage")

if submit:
    try:
        input_df = pd.DataFrame([{
            "crop": crop,
            "soil_type": soil_type,
            "sunlight_hrs": sunlight,
            "temp_c": temp,
            "soil_ph": ph,
            "soil_moisture": moisture,
            "plant_age_days": age,
            "npk_amount": npk,
            "deficiency_score": deficiency
        }])

        # Ensure all columns are present
        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0

        ml = float(model.predict(input_df)[0])
        ng_per_l = ml * 1e6 / 1  # assuming 1L water
        mg_per_l = ml * 1e3 / 1

        st.subheader("💧 Nano Fertilizer Recommendation (conservative)")
        st.metric("Per plant (ml)", f"{ml:.3f}")
        st.metric("Per plant (mg/L)", f"{mg_per_l:.2f}")
        st.metric("Per plant (ng/L)", f"{ng_per_l:.2f}")
        st.write(f"Total for {plants} plants (ml): {ml*plants:.3f}")
        st.warning("⚠️ This recommendation is AI-generated. Always verify with safety guidelines and expert advice.")

    except Exception as e:
        st.error(f"Prediction error: {e}")

# -------------------------
# Upload CSV and Retrain
# -------------------------
st.markdown("---")
st.header("📂 Upload Real Data & Retrain Model")

real_file = st.file_uploader("Upload CSV with same columns as training + nano_amount_ml", type=["csv"])
if real_file:
    try:
        real_df = pd.read_csv(real_file)
        st.dataframe(real_df.head())

        if st.button("Retrain Model with Synthetic + Real Data"):
            synth_df = generate_synthetic_data()
            combined = pd.concat([synth_df, real_df], ignore_index=True)
            model = train_model(combined)
            st.success("✅ Model retrained and saved. Predictions now include real data!")
    except Exception as e:
        st.error(f"Error loading CSV: {e}")

# -------------------------
# Download Synthetic Training Data
# -------------------------
st.markdown("---")
st.download_button(
    "📥 Download Sample Synthetic Training Data",
    generate_synthetic_data(n=500).to_csv(index=False),
    file_name="synthetic_training_data.csv",
    mime="text/csv"
)
