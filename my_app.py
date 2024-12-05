import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# Page configuration
st.set_page_config(page_title="Customer Churn Prediction", layout="wide", page_icon="📊")

# Load the trained model and features
with open("churn_model.pkl", "rb") as file:
    data = pickle.load(file)

if isinstance(data, dict):
    model = data['model']
    model_features = data['features']
else:
    model = data
    model_features = [
        'accountlength', 'numbervmailmessages', 'totaldayminutes', 
        'totaldaycalls', 'totaldaycharge', 'totaleveminutes', 
        'totalevecalls', 'totalevecharge', 'totalnightminutes', 
        'totalnightcalls', 'totalnightcharge', 'totalintlminutes', 
        'totalintlcalls', 'totalintlcharge', 'numbercustomerservicecalls', 
        'internationalplan_yes', 'voicemailplan_yes'
    ]

# Sidebar with logo and title
st.sidebar.image("https://lwfiles.mycourse.app/65a58160c1646a4dce257fac-public/a82c64f84b9bb42db4e72d0d673a50d0.png", use_column_width=True)  # Ensure you have a "logo.png" file in the app directory
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict Churn", "About"])

# About Page
if page == "About":
    st.title("About This App")
    st.markdown("""
        **Customer Churn Prediction** uses machine learning to predict if a customer is likely to churn based on their usage patterns and account details.
        ### Features:
        - Intuitive UI with prediction capabilities.
        - Advanced sidebar navigation.
        - Organized tabs for user inputs and results.
        - Progress bar for real-time status updates.
    """)
    st.info("Developed by [Your Name](https://yourportfolio.com). Powered by Streamlit.")
    st.balloons()
    st.stop()

# Predict Churn Page
st.title("Customer Churn Prediction")
st.markdown("### Enter customer details to predict if they will churn.")

# Input form with tabs
tab1, tab2 = st.tabs(["Input Details", "Result"])

with tab1:
    st.subheader("Customer Information")
    input_data = {
        "accountlength": st.number_input("Account Length", min_value=0, value=120),
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
        "numbercustomerservicecalls": st.number_input("Number of Customer Service Calls", min_value=0, value=2),
        "internationalplan_yes": st.selectbox("International Plan", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"),
        "voicemailplan_yes": st.selectbox("Voicemail Plan", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    }

    if st.button("Predict"):
        with st.spinner("Running prediction..."):
            # Align input with model features
            input_df = pd.DataFrame([input_data])
            input_df = input_df.reindex(columns=model_features, fill_value=0)
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            result = "Churn" if prediction == 1 else "No Churn"

        # Show result in the "Result" tab
        with tab2:
            st.subheader("Prediction Result")
            st.metric(label="Prediction", value=result)
            st.progress(100)
            if result == "Churn":
                st.warning("This customer is predicted to churn.")
            else:
                st.success("This customer is predicted to stay.")

# Footer
st.sidebar.write("---")
st.sidebar.caption("© 2024 [Your Name](https://mdc.edu)")
