import pandas as pd
import datetime as dt
import joblib

# Load the pipeline
def load_pipeline_shoe():
    pipeline_shoe = joblib.load('pipeline_shoe.pkl')
    return pipeline_shoe

def predict_shoe_price(brand, sneaker_name, shoe_size, buyer_region, order_date, release_date, retail_price):
    # Load the pipeline
    pipeline_shoe = load_pipeline_shoe()

    # Create a DataFrame from the input arguments
    input_data = pd.DataFrame({
        'Brand': [brand],
        'Sneaker Name': [sneaker_name],
        'Shoe Size': [shoe_size],
        'Buyer Region': [buyer_region],
        'Order Date': [order_date],  # Ensure the date is in the correct format, e.g., YYYY-MM-DD
        'Release Date': [release_date],  # Same as above
        'Retail Price': [retail_price]  # This will not be used for prediction, but included for consistency
    })

    # Convert 'Order Date' and 'Release Date' to string, then to datetime
    input_data['Order Date'] = pd.to_datetime(input_data['Order Date'])
    input_data['Release Date'] = pd.to_datetime(input_data['Release Date'])

    # Convert the dates to ordinal values
    input_data['Order Date'] = input_data['Order Date'].map(dt.datetime.toordinal)
    input_data['Release Date'] = input_data['Release Date'].map(dt.datetime.toordinal)

    # Use the pipeline to predict the retail price
    predicted_price = pipeline_shoe.predict(input_data)  # Only pass feature columns to the pipeline
    
    return predicted_price[0]
