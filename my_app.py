import streamlit as st
import pandas as pd
import pickle

# Load the trained model and features
with open("churn_model.pkl", "rb") as file:
    data = pickle.load(file)
    model = data['model']
    model_features = data['features']

# Define a function for predictions
def predict_churn(input_data):
    """
    Predict churn based on user input.
    :param input_data: dict with customer data
    :return: str ("Churn" or "No Churn")
    """
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Align input columns with the model's expected features
    input_df = input_df.reindex(columns=model_features, fill_value=0)
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    return "Churn" if prediction == 1 else "No Churn"

# Streamlit UI
st.title("Customer Churn Prediction")
st.write("Input customer data below to predict if they will churn.")

# Collect user input with default values
input_data = {
    "accountlength": st.number_input("Account Length", min_value=0, value=120),
    "internationalplan_yes": st.selectbox("International Plan", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"),
    "voicemailplan_yes": st.selectbox("Voicemail Plan", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"),
    "numbervmailmessages": st.number_input("Number of Voicemail Messages", min_value=0, value=10),
    "totaldayminutes": st.number_input("Total Day Minutes", min_value=0.0, value=250.0),
    "totaldaycalls": st.number_input("Total Day Calls", min_value=0, value=110),
    "totaldaycharge": st.number_input("Total Day Charge", min_value=0.0, value=42.5),
    "totaleveminutes": st.number_input("Total Evening Minutes", min_value=0.0, value=150.0),
    "totalevecalls": st.number_input("Total Evening Calls", min_value=0, value=85),
    "totalevecharge": st.number_input("Total Evening Charge", min_value=0.0, value=20.0),
    "totalnightminutes": st.number_input("Total Night Minutes", min_value=0.0, value=200.0),
    "totalnightcalls": st.number_input("Total Night Calls", min_value=0, value=100),
    "totalnightcharge": st.number_input("Total Night Charge", min_value=0.0, value=9.0),
    "totalintlminutes": st.number_input("Total International Minutes", min_value=0.0, value=15.0),
    "totalintlcalls": st.number_input("Total International Calls", min_value=0, value=4),
    "totalintlcharge": st.number_input("Total International Charge", min_value=0.0, value=3.5),
    "numbercustomerservicecalls": st.number_input("Number of Customer Service Calls", min_value=0, value=2)
}

# Predict button
if st.button("Predict"):
    result = predict_churn(input_data)
    st.write(f"The prediction is: **{result}**")
