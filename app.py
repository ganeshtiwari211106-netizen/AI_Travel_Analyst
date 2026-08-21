import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="AI Travel Analyst", page_icon="✈️", layout="wide")

st.title("✈️ AI Travel Analyst — Flight Price Predictor")
st.markdown("Predict flight ticket prices based on booking parameters using Gradient Boosted Decision Trees.")

# 1. Load Model and Features
@st.cache_resource
def load_artifacts():
    model = joblib.load("saved_models/flight_model.pkl")
    features = joblib.load("saved_models/model_features.pkl")
    return model, features

try:
    model, model_features = load_artifacts()
except Exception as e:
    st.error("Error loading model artifacts. Please run `python src/train_model.py` first.")
    st.stop()

# 2. User Input UI
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Flight & Airline Info")
    airline = st.selectbox("Airline", [
        "Indigo", "Air India", "Vistara", "Airasia India", 
        "Spicejet", "Qatar Airways", "Emirates", "Singapore Airlines", "Etihad Airways"
    ])
    travel_class = st.selectbox("Travel Class", ["Economy", "Premium Economy", "Business", "First"])
    aircraft = st.selectbox("Aircraft Type", [
        "Airbus A320", "Airbus A321", "Airbus A350", "Airbus A380",
        "Boeing 737", "Boeing 777", "Boeing 787 Dreamliner", "Atr 72"
    ])

with col2:
    st.subheader("Route & Timing")
    source = st.selectbox("Source City", ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", "Goa"])
    destination = st.selectbox("Destination City", ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Ahmedabad", "Goa"])
    total_stops = st.selectbox("Total Stops", [0, 1, 2])
    dep_hour = st.slider("Departure Hour (24-Hour Clock)", min_value=0, max_value=23, value=10)

with col3:
    st.subheader("Trip Details")
    distance_km = st.number_input("Distance (in km)", min_value=100.0, max_value=15000.0, value=1200.0, step=50.0)
    duration_hours = st.number_input("Flight Duration (Hours)", min_value=0.5, max_value=30.0, value=2.5, step=0.5)
    days_before = st.slider("Days Before Departure (Booking Window)", min_value=1, max_value=180, value=25)
    season = st.selectbox("Season", ["Summer", "Monsoon", "Autumn", "Winter"])
    weekday = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    booking_channel = st.selectbox("Booking Channel", ["Website", "Mobile App", "Travel Agent", "Third-Party", "Airport Counter"])

# 3. Prediction Execution
if st.button("🔮 Predict Flight Price", use_container_width=True):
    # Construct zeroed DataFrame matching training schema
    input_df = pd.DataFrame(0, index=[0], columns=model_features)
    
    # Assign numerical values
    if "Distance_km" in input_df.columns: input_df.at[0, "Distance_km"] = distance_km
    if "Days_Before_Departure" in input_df.columns: input_df.at[0, "Days_Before_Departure"] = days_before
    if "Duration_Hours" in input_df.columns: input_df.at[0, "Duration_Hours"] = duration_hours
    if "Total_Stops" in input_df.columns: input_df.at[0, "Total_Stops"] = total_stops
    if "Departure_Hour" in input_df.columns: input_df.at[0, "Departure_Hour"] = dep_hour
    
    # Map One-Hot categorical variables
    categorical_inputs = [
        f"Airline_{airline}",
        f"Travel_Class_{travel_class}",
        f"Aircraft_Type_{aircraft}",
        f"Source_{source}",
        f"Destination_{destination}",
        f"Season_{season}",
        f"Weekday_{weekday}",
        f"Booking_Channel_{booking_channel}"
    ]
    
    for feature_col in categorical_inputs:
        if feature_col in input_df.columns:
            input_df.at[0, feature_col] = 1

    # Predict
    predicted_price = model.predict(input_df)[0]
    
    st.markdown("---")
    st.success(f"### Estimated Flight Price: **₹ {predicted_price:,.2f}**")