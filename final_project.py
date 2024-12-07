#!/usr/bin/env python
# coding: utf-8

# # Final Project
# ## Andrew Singh
# ### 12/3/2024

# #### Q1

# #### Read in the data, call the dataframe "s"  and check the dimensions of the dataframe.

# In[116]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import streamlit as st


# In[118]:


st.markdown("# Welcome to my app!")

s = pd.read_csv("social_media_usage.csv")


# In[120]:


s = pd.read_csv("social_media_usage.csv")


# In[122]:


print(s.head())


# In[124]:


print("Dimensions of the dataframe:", s.shape)


# **** 

# #### Q2

# #### Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected

# In[129]:


def clean_sm(x):
    return np.where(x == 1, 1, 0)


# In[131]:


toy_df = {'income': [1,2,3],
          'marital': [1,3,3]}

toy_df = pd.DataFrame(toy_df)

print(toy_df)


# In[133]:


clean_sm(toy_df)


# ****

# #### Q3

# #### Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.

# In[138]:


ss = s[['web1h', 'income', 'educ2', 'par', 'marital', 'gender', 'age']].copy()


# In[140]:


ss.rename(columns={
    'web1h': 'sm_li',
    'educ2': 'education',
    'par': 'parent',
    'marital': 'married',
    'gender': 'female'
}, inplace=True)


# In[142]:


ss['sm_li'] = ss['sm_li'].apply(clean_sm)


# In[144]:


ss['female'] = ss['female'].apply(lambda x: 1 if x == 2 else 0)


# In[146]:


ss = ss[(ss['income'] <= 9) & (ss['education'] <= 8) & (ss['age'] <= 98)].dropna()


# In[148]:


print("Cleaned DataFrame:\n", ss.head())


# In[150]:


ss.dropna(inplace=True)


# In[152]:


print(ss.head())


# In[154]:


print(ss.info())


# In[156]:


print(ss.describe())


# In[158]:


correlation_matrix = ss.corr()
print(correlation_matrix)


# In[160]:


sns.scatterplot(data=ss, x='age', y='income')
plt.show()


# In[162]:


sns.histplot(data=ss, x='age')
plt.show()


# In[164]:


sns.histplot(data=ss, x='income')
plt.show()


# In[166]:


sns.pairplot(ss, hue='income')
plt.show()


# In[167]:


sns.pairplot(ss, hue='sm_li')
plt.show()


# In[168]:


sns.pairplot(ss, hue='education')
plt.show()


# ****

# #### Q4

# #### Create a target vector (y) and feature set (X)

# In[172]:


ss_df = ss


# In[173]:


ss_df = ss_df.drop("sm_li", axis = 1)


# In[174]:


y = ss['sm_li']  
X = ss.drop(columns=['sm_li']) 

print(f"Feature set shape: {X.shape}")
print(f"Target vector shape: {y.shape}")


# ****

# #### Q5

# #### Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning.

# In[178]:


X_train, X_test, y_train, y_test = train_test_split(X.values,
                                                    y, 
                                                    stratify = y,
                                                    test_size=0.2, 
                                                    random_state=42)

print(f"X_train shape: {X_train.shape}") 
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


# Each new object in the model serves a purpose in the machine learning model for this project. 
# 
# X_train contains the feature values we will use for the independent variables in the training dataset. X_train is used in the machine learning model by training the data to learn new patterns between the features and target variable of interest. In this case, X_train contains 80% of the data and contains the features used to predict the target when training the model. 
# 
# X_test contains the portion of the dataset that is not used for training the features component of the dataset, it is used to perform predictions based on our training data. It is used to help compare the predictability of the model compared to Y_test to see how well the model generalizes test data. In this case, X_test contains 20% of the data and contains the features used to test the model on unseen data to evaluate performance. 
# 
# Y_train holds the target values for the dependent variable in the dataset. Y_train is used in training the machine learning algorithm, while supporting a more precise fit between the predictions the model makes and the realistic values in the Y_train dataset. In this case, y_train contains 80% of the data and contains the target that we will predict using the features when training the model. 
# 
# Y_test holds the values for the testing set, this is the portion of the dataset that was not used for training. Y_test is used to make comparison predictions based on the independent values held in X_test to make sure the predictions are not far off. The Y_test data is used to calculate metrics related to evaluating the efficacy of the machine learning model by outputting the precision, recall, and other values. In this case, Y_test contains 20% of the data and contains the target we will predict when testing the model on unseen data to evaluate performance.

# ****

# #### Q6

# #### Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

# In[187]:


logistic_model = LogisticRegression(class_weight='balanced', random_state=42)
logistic_model.fit(X_train, y_train)


# ****

# #### Q7

# #### Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.

# In[359]:


y_pred = logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


# The model accuracy is 0.63.

# In[362]:


pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"])


# The accuracy of the model is .63. 
# 
# In the upper left quadrant, the model correctly predicted that 103 individuals as not being LinkedIn users. 
# In the bottom left quadrant, the model incorrectly predicted 27 LinkedIn users as not actual users. 
# In the upper right quadrant, the model incorrectly predicted 65 individuals as LinkedIn users when they are not.
# In the bottom right quadrant, the model correctly predicted 57 individuals as LinkedIn users. 

# ****

# #### Q8

# #### Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents

# In[368]:


pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")


# **** 

# #### Q9

# #### Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand. Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.

# In[373]:


print(classification_report(y_test, y_pred))


# Accuracy measures the correct number of cases that were identified out of the entire sample. It evaluates how well a model performs overall by comparison of its predicted values to the actual outcomes. An example of this is used to evaluate the efficacy of a model when predicting the number of patients that correctly have a disease out of the entire sample group in a clinical trial.
# 
# Precision is used as a performance metric when the goal is to limit the number of false positives. An example of precision is using a model duirng a clinical trial that is used tp see whether a drug is effective in treating a disease, lets say for example cancer. As trials are very expensive to run and operate the research and development arm of a biotech company might want to run the experiment only if the experiment has a high likelihood of success. This allows the drug to be tested, while in testing the model should not produce as many false positives, it is important that as a result the trial has a high rate of precision or positive predictive value. 
# 
# Recall is measuring how many of the positive samples are captured by the positive predictions. Recall is used as performance metric when we need to identify all positive samples; that is, when it is important to avoid false negatives. An example of when this metric might be the preferred metric for evaluation is when we want to find all the people in a certain trial to be sick, sometimes we may want to find all the patients that are sick and categorize them as positive for a certain disease. In the search for patients that are sick with a certain disease we may categorize those that are not sick as positive for a disease. Even though, this conclusion is false that not all patients are sick, we run the risk of including many false positives in the search for finding patients that are sick. This may damage our recall or sensitivity percentage. 
#                                             
# The F1 score is used to receive the full picture of the situation that combines the precision and recall into one metric. As it takes into account recall and precision, the F1 score provides a better measure than accuracy on imbalanced binary classification datasets. An example of the F1 score being used is when the clinical research team wants to evaluate whether or not a certain patient has a disease or not as the F1 score evaluates the performance of their clinical model being used, they would want to know how many of the correct patient cases are identified as positive cases in this case, the recall metric, and how many are false positives, the precision metric.                                                   
# 
# When calculating the performance metrics by hand, the metrics are equalivant to the produced metrics from the classification report. Metric calculations by are below. 
# 
# Accuracy = TP + TN / Total = 57 + 13 / (103 + 65 + 27 + 53) = .6349 (approx. equal to .63)
# 
# Precision = TP / (TP + FP) = 57 / (57 + 65) = .0.4672 (approx. equal to .47)
# 
# Recall = TP / (TP + FN)= 57 / (57 + 27) = .6786 (approx. equal to .68)
# 
# F1 Score = 2 * (Precision * Recall) / (Precision + Recall) = 2 * (.4672 * .6786) / (.4672 + .6786) = .5516 (approx. equal to .55)

# ****

# #### Q10

# #### Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?
# 

# In[434]:


feature_columns = ['income', 'education', 'parent', 'married', 'female', 'age']


# In[436]:


person_42 = pd.DataFrame([[8, 7, 0, 1, 1, 42]], columns=feature_columns)


# In[438]:


person_82 = pd.DataFrame([[8, 7, 0, 1, 1, 82]], columns=feature_columns)


# In[440]:


if isinstance(X_train, pd.DataFrame):
    prob_high_income_42 = logistic_model.predict_proba(person_42)[0][1]
else:
    prob_high_income_42 = logistic_model.predict_proba(person_42.to_numpy())[0][1]


# In[442]:


if isinstance(X_train, pd.DataFrame):
    prob_high_income_82 = logistic_model.predict_proba(person_82)[0][1]
else:
    prob_high_income_82 = logistic_model.predict_proba(person_82.to_numpy())[0][1]


# In[444]:


print(f"Probability of LinkedIn usage of a 42 years old user): {prob_high_income_42}")


# In[446]:


print(f"Probability of LinkedIn usage of a 82 years old user): {prob_high_income_82}")

