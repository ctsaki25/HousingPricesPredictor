
# Streamlit App for Housing Price Predictor

import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Title
st.title("🏠 Housing Price Predictor")
st.write("Predict house prices based on key features like area, bedrooms, and more!")

# Dataset loading
@st.cache_data
def load_data():
    data_path = "data/house_prices.csv"  # Update if hosting dataset elsewhere
    df = pd.read_csv(data_path)
    return df

df = load_data()

# Preprocessing and model training
def preprocess_and_train_model(df):
    X = df.drop('price', axis=1)
    y = df['price']
    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, mae, mse, r2

model, scaler, mae, mse, r2 = preprocess_and_train_model(df)

# Display model metrics
st.subheader("Model Performance")
st.write(f"**Mean Absolute Error:** {mae:.2f}")
st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# Input fields for prediction
st.subheader("Input Features for Prediction")
area = st.number_input("Area (in sq ft)", min_value=500, step=50)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, step=1)
stories = st.number_input("Number of Stories", min_value=1, step=1)
mainroad = st.selectbox("Main Road Access", ["yes", "no"])
guestroom = st.selectbox("Guest Room Available", ["yes", "no"])
basement = st.selectbox("Basement Available", ["yes", "no"])
hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
parking = st.number_input("Number of Parking Spaces", min_value=0, step=1)
prefarea = st.selectbox("Preferred Area", ["yes", "no"])
furnishingstatus = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

# Prepare input for prediction
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad_yes': [1 if mainroad == "yes" else 0],
        'guestroom_yes': [1 if guestroom == "yes" else 0],
        'basement_yes': [1 if basement == "yes" else 0],
        'hotwaterheating_yes': [1 if hotwaterheating == "yes" else 0],
        'airconditioning_yes': [1 if airconditioning == "yes" else 0],
        'parking': [parking],
        'prefarea_yes': [1 if prefarea == "yes" else 0],
        'furnishingstatus_semi-furnished': [1 if furnishingstatus == "semi-furnished" else 0],
        'furnishingstatus_unfurnished': [1 if furnishingstatus == "unfurnished" else 0]
    })

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"🏠 Predicted House Price: ${prediction:,.2f}")
