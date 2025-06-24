import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import os

# Google Drive downloader for the model file only
def download_file_from_google_drive(file_id, destination):
    if os.path.exists(destination):
        return  # already downloaded
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# Your Google Drive file ID for the random forest model (from your link)
MODEL_FILE_ID = "1iQq39BhysiTvOTey86_C5RojJw3tsbcJ"
MODEL_PATH = "models/random_forest_tuned_model.joblib"

os.makedirs("models", exist_ok=True)
download_file_from_google_drive(MODEL_FILE_ID, MODEL_PATH)

# Load the model and other files locally
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_feature_columns():
    return joblib.load("models/feature_columns.joblib")

@st.cache_resource
def load_neighbourhood_freq_dict():
    return joblib.load("models/neighbourhood_freq_dict.joblib")

model = load_model()
feature_columns = load_feature_columns()
neighbourhood_freq_dict = load_neighbourhood_freq_dict()

st.title("🏠 Airbnb Price Prediction")

neighbourhood = st.selectbox(
    "Select Neighbourhood",
    options=sorted(neighbourhood_freq_dict.keys()),
    index=0,
    help="Choose the neighborhood of the property"
)

neighbourhood_freq = neighbourhood_freq_dict.get(neighbourhood, 0)

host_identity_verified = st.radio(
    "Host Identity Verified?",
    options=[1, 0],
    index=0,
    format_func=lambda x: "Yes" if x == 1 else "No"
)

instant_bookable = st.radio(
    "Instant Bookable?",
    options=[1, 0],
    index=0,
    format_func=lambda x: "Yes" if x == 1 else "No"
)

construction_year = st.number_input(
    "Construction Year",
    min_value=1800,
    max_value=2025,
    value=2000,
    step=1
)

property_age = 2025 - construction_year

minimum_nights = st.number_input(
    "Minimum Nights",
    min_value=1,
    max_value=365,
    value=1,
    step=1
)

number_of_reviews = st.number_input(
    "Number of Reviews",
    min_value=0,
    max_value=10000,
    value=0,
    step=1
)

reviews_per_month = st.number_input(
    "Reviews per Month",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=0.01,
    format="%.2f"
)

review_rate_number = st.slider(
    "Review Rating (1-5)",
    min_value=1,
    max_value=5,
    value=3
)

availability_365 = st.number_input(
    "Availability (days per year)",
    min_value=0,
    max_value=365,
    value=365,
    step=1
)

cancellation_policy = st.selectbox(
    "Cancellation Policy",
    options=["None", "Moderate", "Strict"]
)

cancellation_policy_moderate = cancellation_policy == "Moderate"
cancellation_policy_strict = cancellation_policy == "Strict"

room_type = st.selectbox(
    "Room Type",
    options=["Hotel room", "Private room", "Shared room"]
)

room_type_hotel = room_type == "Hotel room"
room_type_private = room_type == "Private room"
room_type_shared = room_type == "Shared room"

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
    years_ago = days_since_last_review // 365
    months_ago = (days_since_last_review % 365) // 30
    last_review_year = 2025 - years_ago
    last_review_month = 6 - months_ago
    if last_review_month < 1:
        last_review_month += 12
        last_review_year -= 1

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

if st.button("Predict Price"):
    log_price_pred = model.predict(input_df)[0]
    price_pred = np.expm1(log_price_pred)
    st.success(f"Predicted Price: ${price_pred:,.2f}")
