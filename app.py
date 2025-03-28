import streamlit as st
import pandas as pd
import numpy as np
import joblib
import zipfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Extract and Load CSV from ZIP
@st.cache_data
def load_data():
    zip_path = "train.csv.zip"  # Ensure this file exists in GitHub repo
    extract_to = "data"  # Folder where we'll extract CSV

    # Ensure ZIP file exists
    if not os.path.exists(zip_path):
        st.error("ZIP file not found! Make sure 'train.csv.zip' is in your repository.")
        return None

    # Extract ZIP if not already extracted
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

    # Debug: Show extracted files
    extracted_files = os.listdir(extract_to)
    st.write("Extracted files:", extracted_files)  # Debugging

    # Find the CSV file inside the extracted folder
    csv_files = [f for f in extracted_files if f.endswith(".csv")]
    if not csv_files:
        st.error("No CSV file found in the extracted ZIP!")
        return None

    csv_path = os.path.join(extract_to, csv_files[0])  # Use first found CSV
    st.write(f"Using CSV file: {csv_path}")  # Debug print

    # Load CSV file
    df = pd.read_csv(csv_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Remove infinities
    df.dropna(inplace=True)  # Remove NaNs
    return df

df = load_data()
if df is None:
    st.stop()  # Stop execution if data is missing

# Load saved model and selected features
model = joblib.load("gradient_boosting_model.pkl")  # Ensure this is in GitHub repo
selected_features = joblib.load("selected_features.pkl")  # Ensure this is in GitHub repo

st.title("Insurance Loss Prediction 🏦")
st.write("Predict insurance claims using Gradient Boosting")

# Identify categorical and numerical features
categorical_features = df[selected_features].select_dtypes(include=["object"]).columns.tolist()
numerical_features = df[selected_features].select_dtypes(include=["int64", "float64"]).columns.tolist()

# Function to encode categorical features safely
def encode_categorical(df, fit_df):
    encoders = {}
    for col in categorical_features:
        encoders[col] = LabelEncoder()
        fit_df[col] = fit_df[col].astype(str)  # Ensure it's a string
        df[col] = df[col].astype(str)  
        
        encoders[col].fit(fit_df[col])  # Fit on full dataset
        df[col] = df[col].apply(lambda x: x if x in encoders[col].classes_ else "unknown")  # Handle unseen values
        df[col] = encoders[col].transform(df[col])  # Transform input
    return df

# Function to standardize numerical features safely
def standardize_numerical(df, fit_df):
    scaler = StandardScaler()
    fit_df[numerical_features] = scaler.fit_transform(fit_df[numerical_features])  # Fit on full dataset
    df[numerical_features] = scaler.transform(df[numerical_features])  # Transform input
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf values
    df.fillna(df.median(), inplace=True)  # Fill NaNs with median
    return df

# Sidebar for input method
input_method = st.sidebar.radio(
    "Input method:",
    ("Manual Input", "CSV Upload")
)

# Manual Input Mode
if input_method == "Manual Input":
    st.header("Enter Feature Values")

    input_data = {}
    for feature in selected_features:
        if feature in categorical_features:
            unique_values = df[feature].dropna().unique().tolist()
            input_data[feature] = st.selectbox(f"{feature} (category)", unique_values)
        else:
            input_data[feature] = st.number_input(f"{feature}", value=0.0)

    if st.button("Predict"):
        X_new = pd.DataFrame([input_data])

        # Encode categorical features
        X_new = encode_categorical(X_new, df.copy())

        # Standardize numerical features
        X_new = standardize_numerical(X_new, df.copy())

        # Predict
        prediction = np.expm1(model.predict(X_new))[0]

        st.success(f"Predicted Insurance Loss: **${prediction:,.2f}**")

# CSV Upload Mode
else:
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        test_data = pd.read_csv(uploaded_file)

        # Ensure required features are present
        missing_features = [f for f in selected_features if f not in test_data.columns]
        if missing_features:
            st.error(f"Missing features in uploaded file: {missing_features}")
        else:
            X_test = test_data[selected_features]

            # Encode categorical features
            X_test = encode_categorical(X_test, df.copy())

            # Standardize numerical features
            X_test = standardize_numerical(X_test, df.copy())

            # Make predictions
            predictions = np.expm1(model.predict(X_test))

            # Show results
            st.subheader("Predictions")
            results_df = pd.DataFrame({
                "ID": test_data.get("id", range(len(predictions))),
                "Predicted Loss": predictions
            })
            st.dataframe(results_df.style.format({"Predicted Loss": "${:,.2f}"}))

            # Show distribution
            st.subheader("Prediction Distribution")
            fig, ax = plt.subplots()
            sns.histplot(predictions, kde=True, ax=ax)
            ax.set_xlabel("Predicted Loss Amount ($)")
            st.pyplot(fig)
