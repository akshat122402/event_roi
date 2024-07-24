import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load the trained model and scaler
with open('lasso_model.pkl', 'rb') as file:
    lasso = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the input form
st.title('Event Sponsorship ROI Prediction')
st.header('Enter Event Details')

# Input fields for the user
sponsor_costs = st.text_input('Sponsor Costs (separate multiple costs with |)', '1000|2000')
expected_footfall = st.number_input('Expected Footfall', min_value=0, value=100, step=1)
budget = st.number_input('Budget', min_value=0.0, value=1000.0, step=0.01)

# Sponsor types checkboxes
st.text('Sponsor Types:')
financial = st.checkbox('Financial')
food_stalls = st.checkbox('Food Stalls')
in_kind = st.checkbox('In Kind')
influencer = st.checkbox('Influencer')
media_sponsorship = st.checkbox('Media Sponsorship')
merchandise = st.checkbox('Merchandise')
philanthropy = st.checkbox('Philanthropy')

# Event type selection
event_type = st.selectbox('Event Type', [
    'Community and Charity events',
    'Conferences and seminars',
    'Entertainment and media events',
    'Festivals and fairs',
    'Sports events',
    'Virtual event'
])

# When the user clicks the predict button
if st.button('Predict'):
    
    sponsor_costs_mean = np.mean(list(map(float, sponsor_costs.split('|'))))

    # Create a dictionary for sponsor types with binary values
    sponsor_types_dict = {
        'Financial': int(financial),
        'Food Stalls': int(food_stalls),
        'In Kind': int(in_kind),
        'Influencer': int(influencer),
        'Media Sponsorship': int(media_sponsorship),
        'Merchandise': int(merchandise),
        'Philanthropy': int(philanthropy)
    }
    
    # Create a dictionary for event types with binary values
    event_types_dict = {
        'Event Type_Community and Charity events': 0,
        'Event Type_Conferences and seminars': 0,
        'Event Type_Entertainment and media events': 0,
        'Event Type_Festivals and fairs': 0,
        'Event Type_Sports events': 0,
        'Event Type_Virtual event': 0
    }
    event_types_dict[f'Event Type_{event_type}'] = 1

    # Combine all features into a single dictionary
    input_data = {
        'Sponsor Costs': sponsor_costs_mean,
        'Expected Footfall': expected_footfall,
        'Budget': budget
    }
    input_data.update(sponsor_types_dict)
    input_data.update(event_types_dict)
    
    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = lasso.predict(input_scaled)

    roi = (prediction-budget)/budget*100
    st.success(f'Predicted Roi: {roi[0]:.2f}%')










