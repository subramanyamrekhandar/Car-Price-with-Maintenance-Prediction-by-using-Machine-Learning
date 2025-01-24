import streamlit as st
import joblib
import datetime
import pandas as pd
import numpy as np
import random
import time
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pickle

# Set up page config
st.set_page_config(page_title="Unified Prediction App", layout="wide")

# Sidebar Navigation
st.sidebar.title("Select Pipeline")
pipeline = st.sidebar.radio(
    "Choose a prediction module:",
    ("Car Price Prediction", "Car Predictive Maintenance")
)

# Car Price Prediction Pipeline
def car_price_prediction():
    # Load the model
    model = joblib.load('Model/Prediction_Model_Price')

    # Title and description
    st.title("Car Price Prediction App")
    st.markdown("""
    ### Predict the resale value of a car based on its details.
    Fill in the following fields to get an accurate prediction.
    """)

    # Sidebar inputs
    st.sidebar.header("Car Details")
    price = st.sidebar.number_input("Enter Current Price (in Lakhs)", min_value=1.0, max_value=20.0, step=0.1)
    kms = st.sidebar.number_input("Kilometers Driven", min_value=0.0, max_value=30000.0, step=100.0)
    fuel = st.sidebar.selectbox("Fuel Type", ["CNG", "Diesel", "Petrol"])
    seller = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
    mode = st.sidebar.selectbox("Mode Of Transmission", ["Manual", "Automatic"])
    own = st.sidebar.selectbox("No. of Previous Owners", [0, 1, 2, "3+"])
    year = st.sidebar.selectbox("Year of Manufacture", list(range(datetime.datetime.now().year, 1989, -1)))

    # Convert inputs for model
    fuel = {"CNG": 2, "Diesel": 1, "Petrol": 0}[fuel]
    seller = 0 if seller == "Dealer" else 1
    mode = 0 if mode == "Manual" else 1
    age = datetime.datetime.now().year - year
    own = 3 if own == "3+" else own

    # Prediction button
    if st.sidebar.button("Predict"):
        prediction = model.predict([[price, kms, fuel, seller, mode, own, age]])
        final_price = round(prediction[0], 2)
        st.success(f"The predicted resale value of the car is ₹{final_price} Lakhs.")
    else:
        st.info("Enter all the details and click Predict to see the result.")

# Predictive Maintenance Pipeline
def predictive_maintenance():
    # Generate synthetic data to simulate a pre-trained model (if model file doesn't exist)
    def train_and_save_model():
        X, y = make_classification(n_samples=1000, n_features=4, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        with open('Model/predictive_Maintaince_model.pkl', 'wb') as f:
            pickle.dump(model, f)

    # Check if model exists, else create and save one
    try:
        with open('Model/predictive_Maintaince_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        train_and_save_model()
        with open('Model/predictive_Maintaince_model.pkl', 'rb') as f:
            model = pickle.load(f)

    # Generate synthetic sensor data
    def generate_fake_data():
        return {
            'vibration': random.uniform(0.4, 0.8),
            'temperature': random.uniform(50, 100),
            'pressure': random.uniform(25, 35)
        }

    # Predict maintenance needs
    def predict_maintenance(mean_vibration, std_vibration, temp_difference, pressure_diff):
        features = np.array([mean_vibration, std_vibration, temp_difference, pressure_diff]).reshape(1, -1)
        prediction = model.predict(features)
        return bool(prediction[0])

    # Streamlit UI Layout
    st.title("Predictive Maintenance Dashboard")
    st.write("Real-Time Monitoring of Car Sensors with Predictive Maintenance Insights")

    # Real-time sensor data display
    with st.container():
        st.subheader("Real-Time Sensor Data")
        col1, col2, col3 = st.columns(3)
        vibration_display = col1.metric("Vibration (g)", "Fetching...")
        temperature_display = col2.metric("Temperature (°C)", "Fetching...")
        pressure_display = col3.metric("Pressure (psi)", "Fetching...")

    # Real-time graph
    st.subheader("Real-Time Sensor Data Graph")
    graph_placeholder = st.empty()

    # Maintenance prediction status
    st.subheader("Maintenance Prediction")
    prediction_placeholder = st.empty()

    # Main loop for real-time updates
    sensor_data_history = {'time': [], 'vibration': [], 'temperature': [], 'pressure': []}

    # Loop for real-time updates
    for _ in range(20):  # Limit iterations to prevent infinite loop in Streamlit
        sensor_data = generate_fake_data()
        vibration = sensor_data['vibration']
        temperature = sensor_data['temperature']
        pressure = sensor_data['pressure']

        vibration_display.metric("Vibration (g)", f"{vibration:.2f}")
        temperature_display.metric("Temperature (°C)", f"{temperature:.2f}")
        pressure_display.metric("Pressure (psi)", f"{pressure:.2f}")

        prediction_data = {
            'mean_vibration': vibration,
            'std_vibration': random.uniform(0.05, 0.1),
            'temp_difference': random.uniform(0.5, 1.5),
            'pressure_diff': random.uniform(0.5, 1.0)
        }

        maintenance_needed = predict_maintenance(
            prediction_data['mean_vibration'],
            prediction_data['std_vibration'],
            prediction_data['temp_difference'],
            prediction_data['pressure_diff']
        )

        if maintenance_needed:
            prediction_placeholder.error("🚨 Maintenance Needed! ⚠️")
            # prediction_placeholder.success("✅ No Maintenance Needed.")
        else:
            prediction_placeholder.success("✅ No Maintenance Needed.")

        current_time = time.time()
        sensor_data_history['time'].append(current_time)
        sensor_data_history['vibration'].append(vibration)
        sensor_data_history['temperature'].append(temperature)
        sensor_data_history['pressure'].append(pressure)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sensor_data_history['time'], y=sensor_data_history['vibration'], mode='lines+markers', name='Vibration'))
        fig.add_trace(go.Scatter(x=sensor_data_history['time'], y=sensor_data_history['temperature'], mode='lines+markers', name='Temperature'))
        fig.add_trace(go.Scatter(x=sensor_data_history['time'], y=sensor_data_history['pressure'], mode='lines+markers', name='Pressure'))
        graph_placeholder.plotly_chart(fig, use_container_width=True)

        time.sleep(1)

# Run the selected pipeline
if pipeline == "Car Price Prediction":
    car_price_prediction()
elif pipeline == "Car Predictive Maintenance":
    predictive_maintenance()
