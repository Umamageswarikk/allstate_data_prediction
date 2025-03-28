# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# # Load saved model and selected features
# model = joblib.load(r"C:\MCA\placement\digit\venv\Scripts\gradient_boosting_model.pkl")  # Adjust path if necessary
# selected_features = joblib.load(r"C:\MCA\placement\digit\venv\Scripts\selected_features.pkl")

# st.title("Insurance Loss Prediction 🏦")
# st.write("Predict insurance claims using Gradient Boosting")

# # Load dataset for encoding reference
# @st.cache_data
# def load_data():
#     return pd.read_csv(r"C:\MCA\placement\digit\allstate-claims-severity\train.csv")  # Adjust path if necessary

# df = load_data()

# # Identify categorical and numerical features
# categorical_features = df[selected_features].select_dtypes(include=["object"]).columns.tolist()
# numerical_features = df[selected_features].select_dtypes(include=["int64", "float64"]).columns.tolist()

# # Function to encode categorical features
# def encode_categorical(df, fit_df):
#     encoders = {}
#     for col in categorical_features:
#         encoders[col] = LabelEncoder()
#         fit_df[col] = encoders[col].fit_transform(fit_df[col])  # Fit on full dataset
#         df[col] = encoders[col].transform(df[col])  # Transform input
#     return df

# # Function to standardize numerical features
# def standardize_numerical(df, fit_df):
#     scaler = StandardScaler()
#     fit_df[numerical_features] = scaler.fit_transform(fit_df[numerical_features])  # Fit on full dataset
#     df[numerical_features] = scaler.transform(df[numerical_features])  # Transform input
#     return df

# # Sidebar for input method
# input_method = st.sidebar.radio(
#     "Input method:",
#     ("Manual Input", "CSV Upload")
# )

# # Manual Input Mode
# if input_method == "Manual Input":
#     st.header("Enter Feature Values")

#     input_data = {}
#     for feature in selected_features:
#         if feature in categorical_features:
#             unique_values = df[feature].dropna().unique()
#             input_data[feature] = st.selectbox(f"{feature} (category)", unique_values)
#         else:
#             input_data[feature] = st.number_input(f"{feature}", value=0.0)

#     if st.button("Predict"):
#         X_new = pd.DataFrame([input_data])

#         # Encode categorical features
#         X_new = encode_categorical(X_new, df.copy())

#         # Standardize numerical features
#         X_new = standardize_numerical(X_new, df.copy())

#         # Predict
#         prediction = np.expm1(model.predict(X_new))[0]

#         st.success(f"Predicted Insurance Loss: **${prediction:,.2f}**")

# # CSV Upload Mode
# else:
#     st.header("Upload CSV File")
#     uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

#     if uploaded_file:
#         test_data = pd.read_csv(uploaded_file)

#         # Ensure required features are present
#         missing_features = [f for f in selected_features if f not in test_data.columns]
#         if missing_features:
#             st.error(f"Missing features in uploaded file: {missing_features}")
#         else:
#             X_test = test_data[selected_features]

#             # Encode categorical features
#             X_test = encode_categorical(X_test, df.copy())

#             # Standardize numerical features
#             X_test = standardize_numerical(X_test, df.copy())

#             # Make predictions
#             predictions = np.expm1(model.predict(X_test))

#             # Show results
#             st.subheader("Predictions")
#             results_df = pd.DataFrame({
#                 "ID": test_data.get("id", range(len(predictions))),
#                 "Predicted Loss": predictions
#             })
#             st.dataframe(results_df.style.format({"Predicted Loss": "${:,.2f}"}))

#             # Show distribution
#             st.subheader("Prediction Distribution")
#             fig, ax = plt.subplots()
#             sns.histplot(predictions, kde=True, ax=ax)
#             ax.set_xlabel("Predicted Loss Amount ($)")
#             st.pyplot(fig)

# # Model performance section (Validation Data)
# st.markdown("---")
# st.header("Model Performance")

# try:
#     X = df[selected_features]
#     y = df["loss"]  # Adjust if your target column has a different name

#     # Split into train and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Encode categorical features
#     X_train = encode_categorical(X_train, df.copy())
#     X_val = encode_categorical(X_val, df.copy())

#     # Standardize numerical features
#     X_train = standardize_numerical(X_train, df.copy())
#     X_val = standardize_numerical(X_val, df.copy())

#     # Make predictions
#     y_pred = model.predict(X_val)
#     y_true_orig = np.expm1(y_val)
#     y_pred_orig = np.expm1(y_pred)

#     # Compute error metrics
#     mae = mean_absolute_error(y_true_orig, y_pred_orig)
#     rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))

#     col1, col2 = st.columns(2)
#     col1.metric("MAE (Validation)", f"${mae:,.2f}")
#     col2.metric("RMSE (Validation)", f"${rmse:,.2f}")

#     # Actual vs Predicted plot
#     st.subheader("Actual vs Predicted Values")
#     fig, ax = plt.subplots()
#     sns.scatterplot(x=y_true_orig, y=y_pred_orig, alpha=0.3, ax=ax)
#     ax.set_xlabel("Actual Loss")
#     ax.set_ylabel("Predicted Loss")
#     ax.set_title("Actual vs. Predicted Insurance Loss")
#     st.pyplot(fig)

# except Exception as e:
#     st.warning(f"Could not load validation data: {str(e)}")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load saved model and selected features
model = joblib.load("gradient_boosting_model.pkl")  # Adjust path if necessary
selected_features = joblib.load("selected_features.pkl")

st.title("Insurance Loss Prediction 🏦")
st.write("Predict insurance claims using Gradient Boosting")

# Load dataset for encoding reference
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")  # Adjust path if necessary
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Remove infinities
    df.dropna(inplace=True)  # Remove NaNs
    return df

df = load_data()

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
