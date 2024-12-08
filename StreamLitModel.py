#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# #### Streamlit Application Development Code

# In[483]:


st.markdown("#LinkedIn User Prediction Machine")


# In[485]:


st.markdown("#Created by Andrew Singh")


# In[487]:


st.markdown("#Please fill out your information in the respective fields below to predict whether you are a LinkedIn User.")


# In[489]:


st.markdown("#Note: The information used in this form will be held privately and will not be stored.")


# In[491]:


#User input

feature_columns = ['income', 'education', 'parent', 'married', 'female', 'age']

income_options= {
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


# In[493]:


income = st.selectbox("Income Level", options=income_options.keys(), format_func=lambda x: income_options[x])


# In[495]:


education_options= {
    1: "Less than High School",
    2: "High School Incomplete",
    3: "High School Graduate",
    4: "Some College, No Degree",
    5: "Two-Year Associate Degree from a College or University",
    6: "Four-Year College or University Degree/Bachelor's Degree",
    7: "Some Post-Graduate or Professional Schooling",
    8: "Post-Graduate or Professional Degree",
}


# In[497]:


# Education Attainment Level Input
education = st.selectbox(
    "Education Attainment Level",
    options=list(education_options.keys()),
    format_func=lambda x: education_options[x],
)


# In[499]:


parent_options = {
    1: "Yes",
    2: "No",
}


# In[501]:


parent = st.selectbox(
    "Are you a parent?",
    options=list(parent_options.keys()),
    format_func=lambda x: parent_options[x],
)


# In[503]:


marital_options = {
    1: "Yes",
    2: "No",
}


# In[505]:


marital = st.selectbox(
    "Are you married?",
    options=list(marital_options.keys()),
    format_func=lambda x: marital_options[x],
)


# In[507]:


gender_options = {
    1: "Yes",
    2: "No",
}


# In[509]:


gender = st.selectbox(
    "Are you a female?",
    options=list(gender_options.keys()),
    format_func=lambda x: gender_options[x],
)


# In[511]:


age = st.selectbox(
    "Please select your age (1 - 97 years old)",
    options=range(1, 98),  # Range from 1 to 97 (inclusive)
)


# In[513]:


st.write(f"Selected Age: {age}")


# In[515]:


person = [
    user_input["income"],
    user_input["educ2"],
    user_input["par"],
    user_input["marital"],
    user_input["gender"],
    user_input["age"],
]


# In[517]:


# Predict class, given input features
predicted_class = logistic_model.predict([person])[0]

# Generate probability of positive class (=1)
probs = logistic_model.predict_proba([person])[0]


# In[479]:


print(f"Predicted class: {'LinkedIn user' if predicted_class == 1 else 'Not a LinkedIn user'}")

print(f"Probability of being a LinkedIn user: {probs[1]:.2f}")

