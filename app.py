import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Cache model loading so it's done once per session
@st.cache_resource
def load_model():
    return joblib.load("models/random_forest_tuned_model.joblib")

@st.cache_resource
def load_feature_columns():
    return joblib.load("models/feature_columns.joblib")

@st.cache_resource
def load_neighbourhood_freq_dict():
    return joblib.load("models/neighbourhood_freq_dict.joblib")

# Load once
model = load_model()
feature_columns = load_feature_columns()
neighbourhood_freq_dict = load_neighbourhood_freq_dict()

st.title("🏠 Airbnb Price Prediction")

# --- UI inputs ---

# Neighbourhood selectbox with search
neighbourhood = st.selectbox(
    "Select Neighbourhood",
    options=sorted(neighbourhood_freq_dict.keys()),
    index=0,
    help="Choose the neighborhood of the property"
)

# Map neighbourhood to frequency
neighbourhood_freq = neighbourhood_freq_dict.get(neighbourhood, 0)

# Host identity verified
host_identity_verified = st.radio(
    "Host Identity Verified?",
    options=[1, 0],
    index=0,
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# Instant bookable
instant_bookable = st.radio(
    "Instant Bookable?",
    options=[1, 0],
    index=0,
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# Construction Year OR Property Age (only one input)
construction_year = st.number_input(
    "Construction Year",
    min_value=1800,
    max_value=2025,
    value=2000,
    step=1
)

# Calculate property_age = current_year - construction_year
property_age = 2025 - construction_year

# Minimum nights
minimum_nights = st.number_input(
    "Minimum Nights",
    min_value=1,
    max_value=365,
    value=1,
    step=1
)

# Number of reviews
number_of_reviews = st.number_input(
    "Number of Reviews",
    min_value=0,
    max_value=10000,
    value=0,
    step=1
)

# Reviews per month
reviews_per_month = st.number_input(
    "Reviews per Month",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=0.01,
    format="%.2f"
)

# Review rate number (1-5)
review_rate_number = st.slider(
    "Review Rating (1-5)",
    min_value=1,
    max_value=5,
    value=3
)

# Availability 365
availability_365 = st.number_input(
    "Availability (days per year)",
    min_value=0,
    max_value=365,
    value=365,
    step=1
)

# Cancellation policy (Moderate or Strict or Neither)
cancellation_policy = st.selectbox(
    "Cancellation Policy",
    options=["None", "Moderate", "Strict"]
)

cancellation_policy_moderate = cancellation_policy == "Moderate"
cancellation_policy_strict = cancellation_policy == "Strict"

# Room Type
room_type = st.selectbox(
    "Room Type",
    options=["Hotel room", "Private room", "Shared room"]
)

room_type_hotel = room_type == "Hotel room"
room_type_private = room_type == "Private room"
room_type_shared = room_type == "Shared room"

# Last review: choose one method to enter
last_review_option = st.radio(
    "How to input last review date?",
    options=["Year and Month", "Days Since Last Review"]
)

if last_review_option == "Year and Month":
    last_review_year = st.number_input(
        "Last Review Year",
        min_value=1900,
        max_value=2025,
        value=2020,
        step=1
    )
    last_review_month = st.number_input(
        "Last Review Month",
        min_value=1,
        max_value=12,
        value=6,
        step=1
    )
    # Calculate days since last review approx.
    days_since_last_review = (2025 - last_review_year) * 365 + (6 - last_review_month) * 30
    if days_since_last_review < 0:
        days_since_last_review = 0
else:
    days_since_last_review = st.number_input(
        "Days Since Last Review",
        min_value=0,
        max_value=10000,
        value=1000,
        step=1
    )
    # If days_since_last_review given, estimate last_review_year and last_review_month roughly
    years_ago = days_since_last_review // 365
    months_ago = (days_since_last_review % 365) // 30
    last_review_year = 2025 - years_ago
    last_review_month = 6 - months_ago
    if last_review_month < 1:
        last_review_month += 12
        last_review_year -= 1

# --- Prepare input dataframe for model ---

input_dict = {
    "host_identity_verified": host_identity_verified,
    "instant_bookable": instant_bookable,
    "construction_year": construction_year,
    "minimum_nights": minimum_nights,
    "number_of_reviews": number_of_reviews,
    "reviews_per_month": reviews_per_month,
    "review_rate_number": review_rate_number,
    "availability_365": availability_365,
    "cancellation_policy_moderate": cancellation_policy_moderate,
    "cancellation_policy_strict": cancellation_policy_strict,
    "room_type_Hotel room": room_type_hotel,
    "room_type_Private room": room_type_private,
    "room_type_Shared room": room_type_shared,
    "last_review_year": last_review_year,
    "last_review_month": last_review_month,
    "days_since_last_review": days_since_last_review,
    "property_age": property_age,
    "neighbourhood_freq": neighbourhood_freq,
}

input_df = pd.DataFrame([input_dict], columns=feature_columns)

# --- Predict and display result ---

if st.button("Predict Price"):
    # Predict log price
    log_price_pred = model.predict(input_df)[0]
    # Convert back to original price scale
    price_pred = np.expm1(log_price_pred)
    st.success(f"Predicted Price: ${price_pred:,.2f}")


