# Import necessary libraries
import pickle
import datetime as dt
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from cars_notebook import cars_price_predictor  
from banglore_home_prices_final import predict_house_price 
from shoe_price_predictor_function import predict_shoe_price
from laptop_price_predictor import main as laptop_main

# Title of the Streamlit app
st.title("Commodity Price Predictor")

# Create a Landing Page
landing_page = st.sidebar.radio("Navigation", ["Home", "Predict"])

if landing_page == "Home":
    st.image('comm.jpg')
    st.write("""
    Welcome to the Commodity Price Predictor App. This app helps you predict the price of various commodities 
    including Cars, Houses, Laptops, and Shoes.
    """)
    st.write("### How the App Works:")
    st.write("""
    #### This app uses machine learning models to predict the price based on the input data you provide.
    """)
    st.write("### How to Use the App:")
    st.write("""
    1. Use the **sidebar** to select a commodity (Car, House, Laptop, or Shoe).
    2. After selecting, fill in the relevant details like car model, house location, laptop specifications, or shoe brand.
    3. Click on the **'Predict Price'** button to get the price prediction for your chosen commodity.

    Click on 'Predict' in the sidebar to start using the app!
    """)

elif landing_page == "Predict":
    # Sidebar for commodity selection
    commodity = st.sidebar.selectbox("Select Commodity:", ["Car", "House", "Laptop", "Shoe"])

    def load_pipeline_shoe():
        return joblib.load('pipeline_shoe.pkl')

    def load_laptop_data():
        df = joblib.load('df.pkl')
        pipe = joblib.load('pipe.pkl')
        return df, pipe

    if commodity == "Laptop":
        laptop_main()  # Reuse the updated Streamlit UI from laptop_price_predictor.py

    elif commodity == "Car":
        st.image("Car_img.jpg") 
        st.sidebar.header("Car Details")
        company = st.selectbox("Select the car company:", ["Maruti", "Hyundai", "Mahindra", "Tata", "Honda", "Toyota", "Chevrolet", "Renault", "Ford", "Volkswagen", "Skoda", "Audi", "Mini", "BMW", "Datsun", "Mitsubishi", "Nissan", "Mercedes", "Fiat", "Force", "Hindustan", "Jaguar", "Land", "Jeep", "Volvo"])
        year = st.number_input("Enter the year of manufacture:", min_value=2005, max_value=2024)
        kms = st.number_input("Enter the kilometers driven:", min_value=0)
        car_type = st.selectbox("Select the type of car:", ["Petrol", "Diesel", "LPG"])

        if st.button("Predict Car Price"):
            predicted_price = round(int(cars_price_predictor(company, year, kms, car_type)))
            predicted_price = max(1000, min(predicted_price, 1000000))
            st.write(f"Predicted Price for the Car: ₹{predicted_price}")

    elif commodity == "House":
        st.image('house_img.jpg')
        locations_df = pd.read_csv("locations.csv")
        location_columns = locations_df['Location'].tolist()

        st.sidebar.header("House Details")
        location = st.selectbox("Select the Location of Banglore:", location_columns)
        sqft = st.number_input("Enter the area in Square Feet:", min_value=1)
        bath = st.number_input("Enter the number of bathrooms:", min_value=1)
        bhk = st.number_input("Enter the number of BHK (Bedrooms):", min_value=1)

        if st.button("Predict House Price"):
            predicted_price = round(int(predict_house_price(location, sqft, bath, bhk)))
            predicted_price = max(10000, min(predicted_price, 100000000))
            st.write(f"Predicted Price for the House: ₹{predicted_price}")

    elif commodity == "Shoe":
        st.image('shoe_img.jpg', width=450)
        pipeline_shoe = load_pipeline_shoe()

        st.sidebar.header("Shoe Details")
        brand = st.selectbox("Select the Shoe Brand:", [' Yeezy', 'Off-White'])
        sneaker_name = st.selectbox("Select the Sneaker name:", ['Adidas Yeezy Boost', 'Nike Blazer Mid', 'Nike Air Presto','Nike Zoom Fly', 'Nike Air Max', 'Nike Air VaporMax','Nike Air Force', 'Air Jordan 1', 'Nike React Hyperdunk'])
        shoe_size = st.number_input("Enter the Shoe Size:", min_value=1.0, max_value=16.0, step=0.5)
        buyer_region = st.selectbox("Select the Buyer Region:", ['New Jersey', 'New York', 'California', 'Oregon', 'Washington',
           'Idaho', 'Georgia', 'Texas', 'Florida', 'Alabama', 'Pennsylvania',
           'Indiana', 'Virginia', 'North Carolina', 'Nevada', 'Oklahoma',
           'Michigan', 'Arizona', 'New Mexico', 'Maryland', 'Illinois',
           'Massachusetts', 'Ohio', 'Delaware', 'Connecticut', 'Wisconsin',
           'Hawaii', 'Utah', 'Rhode Island', 'Minnesota', 'Missouri',
           'South Carolina', 'Louisiana', 'Colorado', 'District of Columbia',
           'New Hampshire', 'Kansas', 'Kentucky', 'Nebraska', 'West Virginia',
           'Tennessee', 'Arkansas', 'South Dakota', 'Iowa', 'Maine',
           'Wyoming', 'Alaska', 'Mississippi', 'Montana', 'Vermont',
           'North Dakota'])
        order_date = st.date_input("Enter the Order Date:")
        release_date = st.date_input("Enter the Release Date:")
        retail_price = st.number_input("Enter the Retail Price:", min_value=0.0, step=0.01)

        if st.button("Predict Shoe Price"):
            predicted_price = round(predict_shoe_price(brand, sneaker_name, shoe_size, buyer_region, order_date, release_date, retail_price), 2)
            predicted_price = max(50, min(predicted_price, 5000))
            st.write(f"Predicted Price for the Shoe: ${predicted_price}")
