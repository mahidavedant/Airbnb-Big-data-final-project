import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler

# Load the trained model and scaler
with open("./experiments/lin_reg_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

# Create a Streamlit app
st.title("Airbnb Listing Predictor")

# Create input fields with dropdowns, sliders, and text inputs
host_identity_verified = st.selectbox(
    "Host Identity Verified", ["unconfirmed", "verified"])
neighbourhood_group = st.selectbox("Neighbourhood Group", [
                                   "Brooklyn", "Manhattan", "Queens", "Bronx", "Staten Island"])
instant_bookable = st.selectbox("Instant Bookable", ["True", "False"])
cancellation_policy = st.selectbox(
    "Cancellation Policy", ["strict", "moderate", "flexible"])
room_type = st.selectbox(
    "Room Type", ["Private room", "Entire home/apt", "Shared room", "Hotel room"])
construction_year = st.slider("Construction Year", 2003, 2022, 2022)
service_fee = st.slider("Service Fee", 0.0, 240.0, 0.0)
min_nights = st.slider("Min Nights", 1.0, 5645.0, 1.0)
num_of_reviews = st.slider("Num of Reviews", 0.0, 1024.0, 0.0)
reviews_per_month = st.slider("Reviews per Month", 0.01, 90.0, 0.01)
review_rate_number = st.slider("Review Rate Number", 1.0, 5.0, 1.0)
calculated_host_listings_count = st.slider(
    "Calculated Host Listings Count", 1.0, 332.0, 1.0)
availability_365 = st.slider("Availability 365", 0.0, 365.0, 0.0)

# Create a button to make predictions
if st.button("Make Prediction"):
    # Create a dictionary to store the input values
    input_values = {
        "host_identity_verified": [host_identity_verified],
        "neighbourhood_group": [neighbourhood_group],
        "instant_bookable": [1 if instant_bookable == "True" else 0],
        "cancellation_policy": [cancellation_policy],
        "room_type": [room_type],
        "construction_year": [construction_year],
        "service_fee": [service_fee],
        "min_nights": [min_nights],
        "num_of_reviews": [num_of_reviews],
        "reviews_per_month": [reviews_per_month],
        "review_rate_number": [review_rate_number],
        "calculated_host_listings_count": [calculated_host_listings_count],
        "availability_365": [availability_365]
    }

    # Convert dictionary to dataframe
    input_df = pd.DataFrame(input_values)

    # Encode categorical features
    label_encoders = {
        "host_identity_verified": LabelEncoder(),
        "neighbourhood_group": LabelEncoder(),
        "cancellation_policy": LabelEncoder(),
        "room_type": LabelEncoder()
    }

    for feature, encoder in label_encoders.items():
        input_df[feature] = encoder.fit_transform(input_df[feature])

    # Scale features
    scaled_input_df = scaler.fit_transform(input_df)

    # Make predictions using the trained model
    predicted_price = model.predict(scaled_input_df)

    # Display the prediction
    st.write("Prediction:", predicted_price)
