
Commodity Price Predictor 📊💰
==============================

Welcome to the Commodity Price Predictor app! This interactive application lets you predict prices for a variety of commodities, including Cars 🚗, Houses 🏡, Laptops 💻, and Shoes 👟, using machine learning models trained on real-world datasets.



How it Works 🤔
---------------

The app uses different **machine learning algorithms** such as **Linear Regression**, **Random Forest**, and **Random Forest Regressor** to predict the prices of different commodities based on the features provided. The models are trained on historical data sourced from kaggle and are designed to provide reasonably accurate predictions for the given commodities.

### Steps to Use the App 📝

1.  **Select a commodity** from the sidebar:
    *   Car 🚗
    *   House 🏡
    *   Laptop 💻
    *   Shoe 👟
2.  **Provide the necessary details** for the selected commodity (e.g., car model, house location, laptop specifications, or shoe brand).
3.  **Click on the 'Predict Price'** button to get the price prediction for your chosen commodity.
4.  **Get the predicted price** based on the details you provide.

Commodities Available for Prediction 📉
---------------------------------------

### 1\. Car Price Prediction 🚗

This model uses **Linear Regression** to predict the price of a used car based on its features like the **company**, **year of manufacture**, **kilometers driven**, and **fuel type**.

*   **Input Features**: Car company, year of manufacture, kilometers driven, and fuel type (e.g., Petrol, Diesel, LPG).
*   **Model Performance**: The model has an **R² score of 0.78**, indicating that 78% of the variance in car prices can be explained by the features.

### 2\. House Price Prediction 🏡

The **House Price Prediction Model** uses **Linear Regression** to estimate the price of a house based on its **location**, **area**, **number of bathrooms**, and **number of BHK**.

*   **Input Features**: Location, area (in square feet), number of bathrooms, number of BHK (bedrooms).
*   **Model Performance**: The model achieves an **R² score of 0.85**, meaning it explains 85% of the variance in house prices.

### 3\. Laptop Price Prediction 💻

The **Laptop Price Prediction Model** uses the **Random Forest Regressor**, which combines multiple decision trees to predict the price of a laptop based on various specifications like **brand**, **type**, **RAM**, **weight**, and **screen size**.

*   **Input Features**: Brand, type, RAM, weight, touchscreen, screen size, CPU, GPU, HDD, SSD, and operating system.
*   **Model Performance**: The model has an **R² score of 0.89**, indicating it explains 89% of the variance in laptop prices.

### 4\. Shoe Price Prediction 👟

This model uses a **Random Forest Regressor** to predict the resale price of shoes based on features like **brand**, **sneaker name**, **shoe size**, **order date**, **release date**, and **retail price**.

*   **Input Features**: Shoe brand, sneaker name, shoe size, buyer region, order date, release date, and retail price.
*   **Model Performance**: The model has an **R² score of 0.975**, meaning it explains 97.5% of the variance in shoe prices.

How to Use the App 🛠️
----------------------

### 1\. Launch the app

The app will open in your browser where you can interact with the features.

### 2\. Select a Commodity

In the sidebar, select which commodity you want to predict the price for (Car, House, Laptop, or Shoe).

### 3\. Provide the Required Details

Based on the selected commodity, fill in the necessary information such as brand, specifications, or other relevant details.

### 4\. Click on 'Predict Price'

After entering the details, click on the **'Predict Price'** button to get the estimated price.

### 5\. View the Prediction

The predicted price will be displayed on the page. You can also adjust the inputs if you want to try different combinations.

Technologies Used 🧑‍💻
-----------------------

*   **Streamlit**: Used for creating the interactive web interface.
*   **Machine Learning Models**:
    *   **Linear Regression**: For predicting car and house prices.
    *   **Random Forest Regressor**: For predicting laptop and shoe prices.
*   **Python Libraries**:
    *   **Pandas** and **NumPy** for data manipulation.
    *   **Scikit-learn** for model training and predictions.
    *   **Joblib** for loading pre-trained models.

 Project Structure
 -----------------------
 ├── Predictions.py                                             ---> Streamlit App
 
├── cars_notebook.py                                            ---> Car model & logic

├── house_notebook.py                                           ---> House model & logic

├── laptop_notebook.py                                          ---> Laptop model & logic

├── shoe_notebook.py                                            ---> Shoe model & logic

├── pipeline_shoe.pkl                                           ---> Pre-trained shoe model

├── pipe.pkl                                                    ---> Pre-trained laptop model

├── car_model.pkl                                                ---> Pre-trained car model

├── house_model.pkl                                              ---> Pre-trained house model

├── requirements.txt

└── README.md


Contributing 🤝
---------------

We welcome contributions! If you would like to contribute to this project, feel free to fork the repository, make improvements, and submit a pull request.

License 📜
----------

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Thanks for using the **Commodity Price Predictor**! Happy predicting! 🎉

Made By Rakesh Pathlavath
