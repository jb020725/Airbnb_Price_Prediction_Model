# Airbnb Price Prediction Model

This project predicts Airbnb listing prices using a Random Forest regression model trained on cleaned and feature-engineered data. The model leverages key listing features such as neighbourhood, host verification, minimum nights, reviews, and more to generate accurate price estimates.

## Features
- Log-transformed features (`minimum_nights`, `number_of_reviews`) for improved accuracy.
- Neighborhood encoding based on average price.
- Supports categorical inputs like cancellation policy and room type.
- Interactive and user-friendly Streamlit app for price prediction.

## Installation & Usage

```bash
git clone https://github.com/jb020725/Airbnb_Price_Prediction_Model.git
cd Airbnb_Price_Prediction_Model
pip install -r requirements.txt
streamlit run app.py


Open the app in your browser and enter the property details to get a price prediction.

Live Demo
Try the deployed app here:
https://airbnbpricepredictionmodel-jb.streamlit.app/

Project Structure
models/ — Saved model and neighbourhood encoding.

data/cleaned/ — Processed datasets used for training.

app.py — Streamlit app deployment script.

Contact
[janakbhat34@gmail.com](mailto:janakbhat34@gmail.com)



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Made by Janak — Data Science & ML enthusiast.