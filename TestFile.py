#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Mock logistic regression model and scaler (replace with your trained model)
np.random.seed(42)
mock_X = np.random.rand(100, 6)  # Simulated training data with 6 features
mock_y = np.random.randint(0, 2, 100)  # Simulated binary target
scaler = StandardScaler()
mock_X_scaled = scaler.fit_transform(mock_X)
logistic_model = LogisticRegression()
logistic_model.fit(mock_X_scaled, mock_y)

# Application header
st.markdown("# LinkedIn User Prediction Machine")
st.markdown("### Created by Andrew Singh")
st.markdown("### Please fill out your information below to predict whether you are a LinkedIn User.")
st.markdown("#### Note: The information used in this form will be held privately and will not be stored.")

# Feature Options
income_options = {
    1: "Less than $10,000",
    2: "$10,000 to $19,000",
    3: "$20,000 to $29,000",
    4: "$30,000 to $39,000",
    5: "$40,000 to $49,000",
    6: "$50,000 to $74,000",
    7: "$75,000 to $99,000",
    8: "$100,000 to $149,000",
    9: "$150,000 or more",
}

education_options = {
    1: "Less than High School",
    2: "High School Incomplete",
    3: "High School Graduate",
    4: "Some College, No Degree",
    5: "Two-Year Associate Degree from a College or University",
    6: "Four-Year College or University Degree/Bachelor's Degree",
    7: "Some Post-Graduate or Professional Schooling",
    8: "Post-Graduate or Professional Degree",
}

parent_options = {
    1: "Yes",
    2: "No",
}

marital_options = {
    1: "Yes",
    2: "No",
}

gender_options = {
    1: "Yes",
    2: "No",
}

# User Inputs
income = st.selectbox("Income Level", options=list(income_options.keys()), format_func=lambda x: income_options[x])
education = st.selectbox("Education Level", options=list(education_options.keys()), format_func=lambda x: education_options[x])
parent = st.selectbox("Are you a parent?", options=list(parent_options.keys()), format_func=lambda x: parent_options[x])
marital = st.selectbox("Are you married?", options=list(marital_options.keys()), format_func=lambda x: marital_options[x])
gender = st.selectbox("Are you a female?", options=list(gender_options.keys()), format_func=lambda x: gender_options[x])
age = st.slider("Please select your age (1 - 97 years old)", min_value=1, max_value=97)

# Prepare input for prediction
person = [income, education, parent, marital, gender, age]

# Scale the age feature if required
person_scaled = person.copy()
person_scaled[-1] = scaler.transform([[0, 0, 0, 0, 0, person[-1]]])[:, -1][0]

# Predict class and probabilities
predicted_class = logistic_model.predict([person_scaled])[0]
probs = logistic_model.predict_proba([person_scaled])[0]

# Display results
st.write("## Prediction Results")
st.write(f"### Predicted class: {'LinkedIn user' if predicted_class == 1 else 'Not a LinkedIn user'}")
st.write(f"### Probability of being a LinkedIn user: {probs[1]:.2f}")


# In[ ]:




