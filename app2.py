import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model, scaler, and feature names
model = joblib.load('gbm_regressor.pkl')
scaler = joblib.load('scaler2.pkl')
with open('feature_names.pkl', 'rb') as f:
    feature_names = joblib.load(f)

# Define sponsor types and event types
sponsor_types = ["Media Sponsorship", "Food Stalls", "Philanthropy", "Merchandise", "In Kind", "Influencer", "Financial"]
event_types = ["Festivals and fairs", "Virtual event", "Conferences and seminars", "Sports events", "Community and charity events", "Entertainment and media events"]

# Streamlit app
st.title("Sponsor ROI Prediction")

# Input fields for user
event_type = st.selectbox('Event Type', event_types)
sponsor_type = st.selectbox('Sponsor Type', sponsor_types)
sponsor_cost = st.number_input('Sponsor Cost', min_value=1000.0, max_value=200000.0, step=1000.0)
expected_footfall = st.number_input('Expected Footfall', min_value=10, max_value=100000, step=100)
total_num_sponsors = st.number_input('Total Number of Sponsors', min_value=1, max_value=10, step=1)
ticket_price = st.number_input('Ticket Price', min_value=0.0, max_value=500.0, step=1.0)

# Prediction button
if st.button('Predict ROI'):
    # Preprocessing the input
    input_data = {
        'Sponsor Cost': sponsor_cost,
        'Expected Footfall': expected_footfall,
        'Number of Sponsors': total_num_sponsors,
        'Ticket Price': ticket_price
    }

    for sponsor in sponsor_types:
        input_data[f'Has_{sponsor}'] = int(sponsor == sponsor_type)
  
    for event in event_types:
        input_data[f'Is_{event}'] = int(event == event_type)

    input_df = pd.DataFrame([input_data])

    # Ensure the order of columns matches the training set
    input_df = input_df[feature_names]

    # Scale the input features
    input_scaled = scaler.transform(input_df)

    # Predict the ROI
    roi_prediction = model.predict(input_scaled)[0]

    # Display the prediction
    st.success(f'The predicted ROI for the sponsor is: {roi_prediction:.2f}%')
