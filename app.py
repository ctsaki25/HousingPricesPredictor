import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =======================
# Step 1: Model Training
# =======================

@st.cache_data
def load_and_train_models():
    # Load dataset
    data_path = "data/house_prices.csv"  # Ensure this file is uploaded or hosted
    df = pd.read_csv(data_path)

    # Target Transformation
    target_scaler = MinMaxScaler()
    df['scaled_price'] = target_scaler.fit_transform(df[['price']])

    # Feature Engineering
    df['log_area'] = np.log1p(df['area'])
    df['price_per_room'] = df['scaled_price'] / (df['bedrooms'] + 1)
    df['bathrooms_to_bedrooms'] = df['bathrooms'] / (df['bedrooms'] + 1)

    # Outlier Treatment
    for col in ['scaled_price', 'area', 'bedrooms', 'bathrooms']:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = np.clip(df[col], lower, upper)

    # Preprocessing
    X = df.drop(columns=['price', 'scaled_price'])
    y = df['scaled_price']

    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train Models
    catboost_model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=5, verbose=0, random_state=42)
    catboost_model.fit(X_train, y_train)

    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)

    return scaler, target_scaler, X.columns, catboost_model, rf_model, X_test, y_test

# =======================
# Step 2: Evaluation
# =======================
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    st.write(f"**{model_name} Performance:**")
    st.write(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    st.write(f"Mean Squared Error (MSE): ${mse:,.2f}")
    st.write(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    st.write(f"R² Score: {r2:.2f}")

# =======================
# Step 3: Streamlit App
# =======================
st.title("🏠 Housing Price Prediction App")
st.write("Predict the price of a house using CatBoost and Random Forest models.")

# Load and Train Models with Spinner
with st.spinner("Training Models... Please wait!"):
    scaler, target_scaler, feature_names, catboost_model, rf_model, X_test, y_test = load_and_train_models()

# User Inputs
st.write("### Enter House Details")

area = st.number_input("Area (sq. ft)", min_value=500, max_value=10000, value=2000)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=5, value=2)
stories = st.number_input("Number of Stories", min_value=1, max_value=4, value=2)
mainroad = st.selectbox("Main Road Access", ['yes', 'no'])
guestroom = st.selectbox("Guestroom Available", ['yes', 'no'])
basement = st.selectbox("Basement Available", ['yes', 'no'])
hotwaterheating = st.selectbox("Hot Water Heating", ['yes', 'no'])
airconditioning = st.selectbox("Air Conditioning", ['yes', 'no'])
parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=1)
prefarea = st.selectbox("Preferred Area", ['yes', 'no'])
furnishingstatus = st.selectbox("Furnishing Status", ['furnished', 'semi-furnished', 'unfurnished'])

# Button to Predict
if st.button("Predict Price"):
    # Prepare Input Data
    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad_yes': [1 if mainroad == 'yes' else 0],
        'guestroom_yes': [1 if guestroom == 'yes' else 0],
        'basement_yes': [1 if basement == 'yes' else 0],
        'hotwaterheating_yes': [1 if hotwaterheating == 'yes' else 0],
        'airconditioning_yes': [1 if airconditioning == 'yes' else 0],
        'parking': [parking],
        'prefarea_yes': [1 if prefarea == 'yes' else 0],
        'furnishingstatus_semi-furnished': [1 if furnishingstatus == 'semi-furnished' else 0],
        'furnishingstatus_unfurnished': [1 if furnishingstatus == 'unfurnished' else 0],
        'log_area': [np.log1p(area)],
        'price_per_room': [area / (bedrooms + 1)],
        'bathrooms_to_bedrooms': [bathrooms / (bedrooms + 1)]
    })
    input_data = input_data.reindex(columns=feature_names, fill_value=0)
    scaled_input = scaler.transform(input_data)

    # Predict for the given input
    pred_catboost = catboost_model.predict(scaled_input)
    pred_rf = rf_model.predict(scaled_input)

    # Reverse Scaling
    price_catboost = target_scaler.inverse_transform([[pred_catboost[0]]])[0][0]
    price_rf = target_scaler.inverse_transform([[pred_rf[0]]])[0][0]

    # Show Predictions
    st.write(f"### Predicted Prices")
    st.write(f"**CatBoost:** ${price_catboost:,.2f}")
    st.write(f"**Random Forest:** ${price_rf:,.2f}")

    # Evaluate Models Using Test Data
    st.write("### Model Evaluation on Test Data")
    y_pred_catboost_test = catboost_model.predict(X_test)
    y_pred_rf_test = rf_model.predict(X_test)

    evaluate_model(y_test, y_pred_catboost_test, "CatBoost")
    evaluate_model(y_test, y_pred_rf_test, "Random Forest")
