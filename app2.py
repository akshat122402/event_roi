import streamlit as st
import pandas as pd
import joblib
import gzip
import shutil

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Load the model and scaler
model = joblib.load('sponsor_roi_model.pkl')
scaler = joblib.load('scaler1.pkl')

# Function to predict ROI
def predict_roi(input_data):
    # Predict revenue
    predicted_revenue = model.predict(input_data)[0]
    
    # Calculate ROI
    sponsor_cost = input_data['Sponsor Cost'].values[0]
    roi = ((predicted_revenue - sponsor_cost) / sponsor_cost) * 100
    
    # Scale ROI
    scaled_roi = scaler.transform([[roi]])[0][0]
    
    return predicted_revenue, roi, scaled_roi

# Function to categorize scaled ROI
def categorize_roi(scaled_roi):
    if scaled_roi <= 0.208755:
        return "Poor"
    elif scaled_roi <= 0.573525:
        return "Below Average"
    elif scaled_roi <= 1.392291:
        return "Average"
    elif scaled_roi <= 2.0:
        return "Good"
    else:
        return "Excellent"

# Streamlit app
st.title('Sponsor ROI Prediction')

# Input fields
event_type = st.selectbox('Event Type', [
    "Festivals and fairs",
    "Virtual event",
    "Conferences and seminars",
    "Sports events",
    "Community and charity events",
    "Entertainment and media events"
])

event_duration = st.number_input('Event Duration in Days', min_value=1, max_value=7, value=1)
expected_footfall = st.number_input('Expected Footfall', min_value=100, max_value=1000000, value=1000)
ticket_price = st.number_input('Ticket Price', min_value=0, max_value=5000, value=100)

sponsor_type = st.selectbox('Sponsor Type', [
    "Media Sponsorship",
    "Food Stalls",
    "Philanthropy",
    "Merchandise",
    "In Kind",
    "Influencer",
    "Financial"
])

sponsor_cost = st.number_input('Sponsor Cost', min_value=5000, max_value=200000, value=10000)

if st.button('Predict ROI'):
    input_data = pd.DataFrame({
        'Event Type': [event_type],
        'Event Duration in Days': [event_duration],
        'Expected Footfall': [expected_footfall],
        'Ticket Price': [ticket_price],
        'Sponsor Type': [sponsor_type],
        'Sponsor Cost': [sponsor_cost]
    })
    
    predicted_revenue, roi, scaled_roi = predict_roi(input_data)
    roi_category = categorize_roi(scaled_roi)
    
    st.write(f"Predicted ROI Category: {roi_category}")
