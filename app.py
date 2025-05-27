import streamlit as st
import numpy as np
import joblib

# Load the model
model = joblib.load("kmeans_model.h5")

# App title
st.title("Cognitive Insights on Customer Segmentation using watson AI")
st.subheader("Predict which cluster a customer belongs to")

# Input form
annual_income = st.slider("Annual Income (in $1000)", 10, 150, 60)
spending_score = st.slider("Spending Score (1–100)", 1, 100, 50)

# When user clicks "Predict"
if st.button("Predict Cluster"):
    input_data = np.array([[annual_income, spending_score]])
    prediction = model.predict(input_data)
    st.success(f"🧍 This customer belongs to Cluster {prediction[0]}")
