import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

@st.cache_data
def load_data():
    df = pd.read_csv("laptop_data.csv")
    return df

def train_model(df):
    X = df.drop("Price", axis=1)
    y = df["Price"]

    categorical_cols = X.select_dtypes(include="object").columns.tolist()

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ], remainder="passthrough")

    model = Pipeline([
        ("pre", preprocessor),
        ("rf", RandomForestRegressor())
    ])

    model.fit(X, y)
    return model, X

def main():
    st.header("💻 Laptop Price Predictor")

    df = load_data()
    model, X = train_model(df)

    st.subheader("Enter Laptop Details:")
    user_input = {}

    for col in X.columns:
        if df[col].dtype == "object":
            user_input[col] = st.selectbox(f"{col}:", df[col].unique())
        else:
            user_input[col] = st.number_input(f"{col}:", value=float(df[col].mean()))

    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Laptop Price: ₹{int(prediction):,}")

if __name__ == "__main__":
    main()
