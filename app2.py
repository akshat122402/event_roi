import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
import tempfile
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Frame, PageTemplate
from reportlab.lib.styles import getSampleStyleSheet

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

# Function to create PDF with plots
def generate_pdf(plot1_img_path, plot2_img_path, event_type, event_duration, expected_min_footfall, expected_max_footfall, ticket_price, sponsor_type, sponsor_cost, roi_category):
    pdf_filename = "report.pdf"
    buffer = BytesIO()

    # Create a PDF canvas
    c = canvas.Canvas(buffer, pagesize=letter)

    c.setTitle("DATA ANALYSIS")
    c.setFont("Helvetica-Bold", 40)
    c.drawString(160, 725, "DATA ANALYSIS")  # Title

    # Positioning images
    c.setFont("Helvetica", 12)

    # Plot 1
    c.drawImage(plot1_img_path, 72, 460, width=500, height=250)  # Adjust size and position for Plot 1

    # Plot 2
    c.drawImage(plot2_img_path, 72, 150, width=500, height=250)  # Adjust size and position for Plot 2

    c.showPage()

    c.setTitle("ROI")
    c.setFont("Helvetica-Bold", 40)
    c.drawString(120, 725, "ROI CALCULATIONS") 

    c.drawImage('speedometer.png', 50, 580, width=160, height=90)  # Speedometer image
    
    c.setFont("Helvetica", 18)
    c.drawString(310, 675, "Expected ROI Category")

    # Create a Paragraph with the long string
    str_roi1 = "The expected ROI for the Sponsor is calculated based on"
    str_roi2 = "the costs incurred and the revenue generated. With an" 
    str_roi3 = f"investment of Rs. {sponsor_cost} in sponsorship and advertising, "
    str_roi4 = f"the expected revenue is estimated to be '{roi_category}'."
    
    c.setFont("Helvetica", 12)
    c.drawString(250, 640, str_roi1)
    c.drawString(250, 625, str_roi2)
    c.drawString(250, 610, str_roi3)
    c.drawString(250, 595, str_roi4)
    

    c.save()

    # Save the pdf file in memory
    buffer.seek(0)  # Return to start of the BytesIO buffer
    return buffer

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
expected_min_footfall = st.number_input('Expected Minimum Footfall', min_value=100, max_value=1000000, value=1000)
expected_max_footfall = st.number_input('Expected Maximum Footfall', min_value=100, max_value=1000000, value=10000)
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
        'Expected Footfall': [(expected_max_footfall + expected_min_footfall) / 2],
        'Ticket Price': [ticket_price],
        'Sponsor Type': [sponsor_type],
        'Sponsor Cost': [sponsor_cost]
    })

    predicted_revenue, roi, scaled_roi = predict_roi(input_data)
    roi_category = categorize_roi(scaled_roi)

    st.write(f"Predicted ROI Category: {roi_category}")

    # Automatically generate the plots
    sponsor_types = [
        "Media Sponsorship", "Food Stalls", "Philanthropy", 
        "Merchandise", "In Kind", "Influencer", "Financial"
    ]
    
    roi_categories = []
    for s_type in sponsor_types:
        temp_input = input_data.copy()
        temp_input['Sponsor Type'] = s_type
        _, _, temp_scaled_roi = predict_roi(temp_input)
        roi_categories.append(categorize_roi(temp_scaled_roi))

    # Plot 1: ROI Category vs Sponsor Type
    category_order = ["Poor", "Below Average", "Average", "Good", "Excellent"]
    roi_categories_numeric = [category_order.index(cat) for cat in roi_categories]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=sponsor_types, y=roi_categories_numeric, order=sponsor_types)
    plt.title("ROI Category vs Sponsor Type")
    plt.xlabel("Sponsor Type")
    plt.ylabel("ROI Category")
    plt.yticks(range(len(category_order)), category_order)
    
    # Save Plot 1 to a temporary image file
    plot1_img_path = tempfile.mktemp(suffix=".png")
    plt.savefig(plot1_img_path, format='png')  
    plt.close()

    # Prepare for Plot 2: ROI Category vs Sponsor Cost
    sponsor_cost_range = np.arange(max(sponsor_cost - 100000, 10000), sponsor_cost + 100000, 10000)
    roi_categories_cost = []

    for cost in sponsor_cost_range:
        temp_input = input_data.copy()
        temp_input['Sponsor Cost'] = cost
        _, _, temp_scaled_roi = predict_roi(temp_input)
        roi_categories_cost.append(categorize_roi(temp_scaled_roi))

    # Convert categories to numeric values for plotting
    category_labels = sorted(set(roi_categories_cost), key=lambda x: category_order.index(x))
    category_to_num = {cat: i for i, cat in enumerate(category_labels)}
    y_numeric = [category_to_num[cat] for cat in roi_categories_cost]

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(sponsor_cost_range, y_numeric, marker='o')

    # Add horizontal lines for each category
    for i, category in enumerate(category_labels):
        plt.hlines(y=i, xmin=sponsor_cost_range[0], xmax=sponsor_cost_range[-1], colors='blue', linestyles='dashed', alpha=0.5)

    plt.title("ROI Category vs Sponsor Cost")
    plt.xlabel("Sponsor Cost")
    plt.ylabel("ROI Category")
    plt.yticks(list(category_to_num.values()), category_labels)  
    plt.xticks(sponsor_cost_range, rotation=45)  
    plt.grid()

    # Save Plot 2 to a temporary image file
    plot2_img_path = tempfile.mktemp(suffix=".png")
    plt.savefig(plot2_img_path, format='png')  
    plt.close()

    # Create a PDF with the plots
    pdf_buffer = generate_pdf(plot1_img_path, plot2_img_path, event_type, event_duration, expected_min_footfall, expected_max_footfall, ticket_price, sponsor_type, sponsor_cost, roi_category)

    # Provide a download link for the PDF
    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name="report.pdf",
        mime="application/pdf"
    )

    # Optionally delete the temporary image files
    os.remove(plot1_img_path)
    os.remove(plot2_img_path)
