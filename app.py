import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model and neighborhood encoder
@st.cache_resource
def load_model():
    return joblib.load("models/random_forest_model.pkl")

@st.cache_resource
def load_neighbourhood_encoder():
    return joblib.load("models/neighbourhood_encoding.pkl")

model = load_model()
neighbourhood_encoder = load_neighbourhood_encoder()

st.title("🏠 Airbnb Price Prediction")

# User inputs
neighbourhood = st.selectbox(
    "Select Neighbourhood",
    options=sorted(neighbourhood_encoder.keys())
)

host_identity_verified = st.radio(
    "Host Identity Verified?",
    options=[1, 0],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

instant_bookable = st.radio(
    "Instant Bookable?",
    options=[1, 0],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

construction_year = st.number_input(
    "Construction Year", min_value=1800, max_value=2025, value=2015, step=1
)
building_age = 2025 - construction_year

minimum_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, value=1, step=1)
number_of_reviews = st.number_input("Number of Reviews", min_value=0, value=0, step=1)
reviews_per_month = st.number_input("Reviews per Month", min_value=0.0, value=0.0, step=0.01)
review_rate_number = st.slider("Review Rating (1-5)", min_value=1, max_value=5, value=3)
availability_365 = st.number_input("Availability (days/year)", min_value=0, max_value=365, value=180)

days_since_last_review = st.number_input("Days Since Last Review", min_value=0, value=1000, step=1)

cancellation_policy = st.selectbox("Cancellation Policy", options=["None", "Moderate", "Strict"])
cancellation_policy_moderate = cancellation_policy == "Moderate"
cancellation_policy_strict = cancellation_policy == "Strict"

room_type = st.selectbox("Room Type", options=["Hotel room", "Private room", "Shared room"])
room_type_hotel = room_type == "Hotel room"
room_type_private = room_type == "Private room"
room_type_shared = room_type == "Shared room"

neighbourhood_encoded = neighbourhood_encoder.get(neighbourhood, neighbourhood_encoder.mean())

input_data = [
    reviews_per_month,
    review_rate_number,
    availability_365,
    neighbourhood_encoded,
    bool(host_identity_verified),
    bool(instant_bookable),
    cancellation_policy_moderate,
    cancellation_policy_strict,
    room_type_hotel,
    room_type_private,
    room_type_shared,
    days_since_last_review,
    building_age,
    np.log1p(minimum_nights),
    np.log1p(number_of_reviews),
]

# Convert to DataFrame with columns in correct order
input_df = pd.DataFrame([input_data], columns=[
    'reviews_per_month',
    'review_rate_number',
    'availability_365',
    'neighbourhood_encoded',
    'host_identity_verified_verified',
    'instant_bookable_True',
    'cancellation_policy_moderate',
    'cancellation_policy_strict',
    'room_type_Hotel room',
    'room_type_Private room',
    'room_type_Shared room',
    'days_since_last_review',
    'building_age',
    'minimum_nights',
    'number_of_reviews'
])

# Predict
if st.button("Predict Price"):
    price_pred = model.predict(input_df)[0]
    st.success(f"Predicted Airbnb Price: ${price_pred:,.2f}")

