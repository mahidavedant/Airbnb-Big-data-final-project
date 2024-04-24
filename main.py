import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the trained model
with open('./experiments/lin_reg_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a Streamlit app
st.title("Airbnb Price Predictor")

# Create a form with input fields
form = st.form(key="airbnb-predict-form")

with form:
    # Add input fields with hints and help
    host_identity_verified = st.selectbox("Host Identity Verified", [
                                          "unconfirmed", "verified"], help="Select host identity verification status")
    neighbourhood_group = st.selectbox("Neighbourhood Group", [
                                       "Brooklyn", "Manhattan", "Queens", "Bronx", "Staten Island"], help="Select neighbourhood group")
    instant_bookable = st.checkbox(
        "Instant Bookable", help="Check if instant booking is available")
    cancellation_policy = st.selectbox("Cancellation Policy", [
                                       "strict", "moderate", "flexible"], help="Select cancellation policy")
    room_type = st.selectbox("Room Type", [
                             "Private room", "Entire home/apt", "Shared room", "Hotel room"], help="Select room type")
    construction_year = st.number_input("Construction Year", min_value=2003,
                                        max_value=2022, value=2010, help="Enter construction year (2003-2022)")
    service_fee = st.number_input(
        "Service Fee", min_value=0.0, max_value=240.0, value=20.0, help="Enter service fee (0-240)")
    min_nights = st.number_input("Minimum Nights", min_value=1.0,
                                 max_value=5645.0, value=30.0, help="Enter minimum nights (1-5645)")
    num_of_reviews = st.number_input("Number of Reviews", min_value=0.0,
                                     max_value=1024.0, value=50.0, help="Enter number of reviews (0-1024)")
    reviews_per_month = st.number_input(
        "Reviews per Month", min_value=0.01, max_value=90.0, value=5.0, help="Enter reviews per month (0.01-90)")
    review_rate_number = st.number_input(
        "Review Rate Number", min_value=1.0, max_value=5.0, value=4.0, help="Enter review rate number (1-5)")
    calculated_host_listings_count = st.number_input(
        "Calculated Host Listings Count", min_value=1.0, max_value=332.0, value=100.0, help="Enter calculated host listings count (1-332)")
    availability_365 = st.number_input(
        "Availability (365 days)", min_value=0.0, max_value=365.0, value=180.0, help="Enter availability (0-365)")

    # Add a submit button
    submitted = st.form_submit_button("Predict Price")

# Create a container for the prediction output
output_container = st.container()

if submitted:
    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'host_identity_verified': [host_identity_verified],
        'neighbourhood_group': [neighbourhood_group],
        'instant_bookable': [instant_bookable],
        'cancellation_policy': [cancellation_policy],
        'room_type': [room_type],
        'construction_year': [construction_year],
        'service_fee': [service_fee],
        'min_nights': [min_nights],
        'num_of_reviews': [num_of_reviews],
        'reviews_per_month': [reviews_per_month],
        'review_rate_number': [review_rate_number],
        'calculated_host_listings_count': [calculated_host_listings_count],
        'availability_365': [availability_365]
    })

    # Create a pipeline with LabelEncoder, RobustScaler, and LinearRegression
    pipeline = Pipeline([
        ('encoder', LabelEncoder()),
        ('scaler', RobustScaler()),
        ('model', LinearRegression())
    ])

    # Fit the pipeline to the input data
    pipeline.fit(input_data)

    # Make a prediction using the pipeline
    prediction = pipeline.predict(input_data)

    # Display the predicted price
    with output_container:
        st.write(f"Predicted Price: ${prediction[0]:.2f}")
