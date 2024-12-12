#!/usr/bin/env python
# coding: utf-8

# In[18]:


import streamlit as st
import pandas as pd
from JobLib import get_trained_model  # Import the model function

# Load the model
try:
    model = get_trained_model()  # Load model directly from JobLib.py
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    model = None

# Page layout
st.title("LinkedIn User Prediction Application")
st.markdown("### Created by Andrew Singh")
st.markdown("Please fill out your information to predict your likelihood of being a LinkedIn user.")
st.markdown("**Note:** Your information will not be stored or shared.")

# Dropdown options
income_options = {
    "Less than $10,000": 1,
    "$10,000 to $19,999": 2,
    "$20,000 to $29,999": 3,
    "$30,000 to $39,999": 4,
    "$40,000 to $49,999": 5,
    "$50,000 to $74,999": 6,
    "$75,000 to $99,999": 7,
    "$100,000 to $150,000": 8,
    "Greater than $150,000": 9,
}

education_options = {
    "Less than high school": 1,
    "High school incomplete": 2,
    "High school graduate": 3,
    "Some college, no degree": 4,
    "Two-year associate degree": 5,
    "Four-year college/university degree": 6,
    "Some postgraduate or professional schooling": 7,
    "Postgraduate or professional degree": 8,
}

# Input form
with st.form("prediction_form"):
    st.subheader("Demographic Information")
    income = st.selectbox("What is your income level?", options=list(income_options.keys()))
    education = st.selectbox("What is your education level?", options=list(education_options.keys()))
    age = st.slider("Use the slider to input your age:", min_value=1, max_value=98, value=25)
    parent = st.radio("Are you a parent?", ["No", "Yes"])
    married = st.radio("Are you married?", ["No", "Yes"])
    female = st.radio("Do you identify as a female?", ["No", "Yes"])

    submit_button = st.form_submit_button("Submit")

# Prediction logic
if submit_button:
    if model is None:
        st.error("Model is not loaded. Unable to make predictions.")
    else:
        # Prepare input data
        features = pd.DataFrame(
            [[
                income_options[income],
                education_options[education],
                age,
                1 if parent == "Yes" else 0,
                1 if married == "Yes" else 0,
                1 if female == "Yes" else 0
            ]],
            columns=['income', 'education','age','parent', 'married', 'female']  # Correct column names and order
        )

        try:
            # Make predictions
            prediction = model.predict(features)
            probability = model.predict_proba(features)[0][1] * 100  # Convert to percentage

            # Display results
            st.header("Prediction Results")
            st.metric("Predicted Category", "LinkedIn User" if prediction[0] == 1 else "Not a LinkedIn User")
            st.metric("Probability of being a LinkedIn User", f"{probability:.1f}%")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

