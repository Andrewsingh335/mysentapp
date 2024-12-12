#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_trained_model():
    # Load dataset
    s = pd.read_csv("social_media_usage.csv")

    def clean_sm(x):
        return np.where(x == 1, 1, 0)

    # Preprocess data
    ss = s[['web1h', 'income', 'educ2', 'par', 'marital', 'gender', 'age']].copy()
    ss.rename(columns={
        'web1h': 'sm_li',       # Target variable
        'educ2': 'education',   # Education feature
        'par': 'parent',        # Parent status feature
        'marital': 'married',   # Marital status feature
        'gender': 'female'      # Gender feature
    }, inplace=True)
    ss['sm_li'] = ss['sm_li'].apply(clean_sm)
    ss['female'] = ss['female'].apply(lambda x: 1 if x == 2 else 0)
    ss = ss[(ss['income'] <= 9) & (ss['education'] <= 8) & (ss['age'] <= 98)].dropna()

    # Ensure feature order matches Streamlit app
    feature_columns = ['income', 'education', 'age', 'parent', 'married', 'female']
    X = ss[feature_columns]
    y = ss['sm_li']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    return model



