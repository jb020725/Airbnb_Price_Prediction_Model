# Airbnb Price Prediction Model

This project predicts Airbnb listing prices using a Random Forest regression model trained on cleaned and feature-engineered data. The model uses key listing features such as neighbourhood, host verification, minimum nights, reviews, and more to generate price estimates.

## Features
- Uses log-transformed features (`minimum_nights`, `number_of_reviews`) for better model accuracy.
- Encodes neighbourhoods by average price.
- Supports categorical inputs like cancellation policy and room type.
- User-friendly Streamlit app for interactive price prediction.

## Installation

```bash
git clone https://github.com/jb020725/Airbnb_Price_Prediction_Model.git
cd Airbnb_Price_Prediction_Model
pip install -r requirements.txt
streamlit run app.py


Usage
Open the Streamlit app and enter the property details to get an estimated price.

Live Demo
Try the deployed app here:
https://airbnbpricepredictionmodel-jb.streamlit.app/

Files
models/ — Saved model and neighbourhood encoding.

data/cleaned/ — Processed datasets used for training.

app.py — Streamlit deployment code.

Contact
For questions or collaborations, reach out at-
[janakbhat34@gmail.com](mailto:janakbhat34@gmail.com)


License
This project is open source and free to use.

Made by Janak — Data Science & ML enthusiast.