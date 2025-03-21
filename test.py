import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
with open('train_delay_model.pkl', 'rb') as f:
    model_pipeline = joblib.load(f)

st.title("Train Delay Predictor")

st.markdown("Enter the train details to predict the delay in minutes.")


train_id = st.number_input("Train ID", value=10000, step=1)

schedule_departure = st.text_input("Schedule Departure (YYYY-MM-DD HH:MM:SS)", value="2025-03-31 23:17:00")


delay_reasons = ["signal failure", "track maintenance", "bad weather",
                 "train pass on", "accidents", "congestion", "chain pull", "emergency stop"]
reason = st.selectbox("Delay Reason", delay_reasons)

if st.button("Predict Delay"):
    try:
        sd = pd.to_datetime(schedule_departure)
        dep_hour = sd.hour
        dep_minute = sd.minute
        dep_dayofweek = sd.dayofweek

        input_data = pd.DataFrame({
            'train_id': [train_id],
            'dep_hour': [dep_hour],
            'dep_minute': [dep_minute],
            'dep_dayofweek': [dep_dayofweek],
            'reason': [reason]
        })
        prediction = model_pipeline.predict(input_data)
        st.success(f"Predicted Delay: {prediction[0]:.2f} minutes")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
