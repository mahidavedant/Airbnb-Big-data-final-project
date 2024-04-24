import pandas as pd
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# host_identity_verified -  "unconfirmed", "varified"
# neighbourhood_group - 'Brooklyn' 'Manhattan' 'Queens' 'Bronx' 'Staten Island'
# instant_bookable - True False
# cancellation_policy - 'strict' 'moderate' 'flexible'
# room_type - 'Private room' 'Entire home/apt' 'Shared room' 'Hotel room'
# construction year - 2003 ... 2022
# service_fee: 0.0 ... 240.0
# min_nights - 1.0 ... 5645.0
# num_of_reviews - 0.0 ... 1024.0
# reviews_per_month - 0.01 ... 90.0
# review_rate_number - 1.0 ... 5.0
# calculated_host_listings_count - 1.0, 332.0
# availaibility_365 - 0.0 ... 365

with open("./experiments/lin_reg_model.pkl", 'rb') as f:
    model = pickle.load(f)


# Create a LabelEncoder object
le = LabelEncoder()


st.title("AirBnb Price Predictor🏠")

with st.form(key="ml_input_form"):
    host_identity_verified = st.selectbox(
        "Host Identity Verified", ["unconfirmed", "verified"])
    neighbourhood_group = st.selectbox("Neighbourhood Group", [
                                       'Brooklyn', 'Manhattan', 'Queens', 'Bronx', 'Staten Island'])
    instant_bookable = st.checkbox("Instant Bookable")
    cancellation_policy = st.selectbox(
        "Cancellation Policy", ['strict', 'moderate', 'flexible'])
    
    room_type = st.selectbox(
        "Room Type", ['Private room', 'Entire home/apt', 'Shared room', 'Hotel room'])
    
    construction_year = st.number_input(
        "Construction Year", min_value=2003, max_value=2022)
    
    service_fee = st.number_input("Service Fee")
    min_nights = st.number_input("Minimum Nights")
    calculated_host_listings_count = st.number_input(
        "Calculated Host Listings Count")
    price_per_night = st.number_input("Price per Night")

    submit_button = st.form_submit_button("Predict")


if submit_button:

    # Convert categorical values to numerical representations using LabelEncoder
    host_identity_verified = le.fit_transform([host_identity_verified])[0]
    neighbourhood_group = le.fit_transform([neighbourhood_group])[0]
    cancellation_policy = le.fit_transform([cancellation_policy])[0]
    room_type = le.fit_transform([room_type])[0]

    # Create a numpy array from the input values
    input_values = np.array([host_identity_verified, neighbourhood_group, instant_bookable, cancellation_policy, room_type,
                            construction_year, service_fee, min_nights, calculated_host_listings_count, price_per_night])

    # Make a prediction using the model
    prediction = model.predict([input_values])

    # Display the prediction
    st.write("Prediction:", prediction[0])
