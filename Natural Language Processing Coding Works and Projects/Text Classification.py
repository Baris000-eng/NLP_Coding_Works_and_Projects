#!/usr/bin/env python
# coding: utf-8

# In[127]:


# Machine Learning
# What is Machine Learning ?
# Machine Learning (ML) is a method of data analysis that automates analytical model building.
# Using alogirthms that iteratively learn from data, machine learning allows computers to find
# the hidden insights without being explicitly programmed where to look.
# What is machine learning used for ? (In which areas does it take place ?)
# 1-) Recommendation Engines
# 2-) Customer Segmentation
# 3-) Text Sentiment Analysis
# 4-) Prediction of the Customer Churn
# 5-) Email Spam Filtering
# 6-) Image and Pattern Recognition
# 7-) Network Intrusion Detection
# 8-) Price Models 

# * Supervised Learning:
# * Supervised Learning algorithms are trained using labeled examples, like an input where the 
# desired output is known.
# For instance, a segment of the text could have a categorical label , such as: 
# * Positive vs Negative Movie Review
# * Spam vs Legitimate Email

# The supervised machine learning algorithm receives a set of inputs along with the corresponding correct 
# outputs, and this algorithm learns by comparing its actual output with correct outputs to 
# find errors.

# Supervised learning is commonly used in applications where historical data predicts likely future 
# events. As an example to the historical data, we can give the historical information of emails 
# received that someone has labeled spam or legitimate.

# Supervised Machine Learning Process:
# 1-) Data Acquisition (Data Gathering)
# 2-) Data Cleaning and Formatting (In this step, Scikit-Learn and Vectorization are widely used for the text data.)
# 3-) Splitting the data into the training set and test set
# 4-) Model Training & Building (fitting the model to the training data)
# 5-) Model Testing (by using the test dataset)
# 6-) Model Deployment

# Text classification and recognition is a very common and widely applicable use of machine learning.

# * Before the data split, we have labels and features.
#   - Features: x
#   - Labels: y

# * Before we fit the model, we split the data into training dataset and test dataset.

# After the train-test split, we always have the 4 components below:
# 1-) X_train
# 2-) X_test
# 3-) Y_train
# 4-) Y_test


# Classification Evaluation Metrics

# The key classification metrics are as below:
# 1-) Accuracy
# 2-) Precision
# 3-) Recall
# 4-) F1-Score

# In a supervised learning algorithm, we will first train/fit a model on the training data, and then test the model
# on the testing data. Once we have the model's predictions from the X_test data, we compare them to the true y 
# values (the correct labels).


"""xtrain (X_train):

"xtrain" typically refers to the feature or input data that is used for training a machine learning model. 
It is a subset of your entire dataset.This dataset contains the input variables or features that the model 
will use to make predictions. Each row in the "xtrain" corresponds to one data point, and each column 
represents a different feature or attribute of the data.

ytrain (y_train):

"ytrain" corresponds to the target or output values that are associated with the training data. These target 
values are known as labels or ground truth. In supervised learning, the goal is for the model to learn a mapping 
from the input data (xtrain) to the target values (ytrain) during the training phase. It learns to make predictions 
based on this labeled data.

xtest (X_test):

"xtest" is another subset of your dataset, distinct from the training data. It contains the feature or input data 
that you use for testing the trained model's performance.The model will make predictions on this test data to 
assess how well it generalizes to unseen examples.

ytest (y_test):

"ytest" corresponds to the target or output values associated with the test data. These are the correct labels 
for the test examples. During evaluation, you compare the model's predictions on "xtest" to the actual values 
in "ytest" to measure its performance. This helps you understand how well the model can make predictions on 
new, unseen data."""

# We could place the real values and our predicted values in a confusion matrix.

# Accuracy: 
# Accuracy in classification problems is the number of correct predictions made by the model divided by the 
# total number of predictions (i.e. Accuracy = number of correct predictions / total number of predictions)

# Accuracy Formula = ((TP + TN) / (TP + TN + FP + FN))

# Accuracy is useful when target classes are well balanced. However, accuracy is not a good choice for the 
# unbalanced classes.


# Precision:
# It is the ability of a classification model to identify only the relevant data points. Precision is defined
# as the number of true positives divided by the number of true positives plus the number of false 
# positives (Precision = TP / (TP + FP)).

# Recall: 
# It is the ability of a model to find all the relevant cases within a dataset.
# The definition of recall is the number of true positives divided by the number of true positives plus 
# the number of false negatives (Recall = TP / (TP + FN)).

# F1-Score:
# The F1-Score is the harmonic mean of precision and recall. It takes both metrics into account in the 
# following equation.

# F1-Score = (2 * Precision * Recall) / Precision + Recall
# OR
# F1-Score = 2 / ((1 / Precision) + (1 / Recall))

# In F1-Score, harmonic mean instead of a simple average is used to punish extreme values (e.g precision or 
# recall being 0).


#-------------------------------------------------------------------------------------------------------------#

# Confusion Matrix

# Confusion matrix is a way to view various classification metrics.

# Note: In a classification problem, during the testing phase, we will have two following categories:

# 1-) True Condition: For example, a text message is spam.
# 2-) Predicted Condition: For example, ML model predicted that the text message is spam.

# This means when we have two possible classes, it should bring up 4 seperate groups at the end of testing. 

# Note: Here, class 1 is HAM and class 2 is SPAM.

# Correctly classified to the class 1: TRUE HAM
# Correctly classified to the class 2 : TRUE SPAM
# Incorrectly classified to the class 1 : FALSE HAM
# Incorrectly classified to the class 2 : FALSE SPAM


# Real condition is HAM and the prediction is HAM : True Positive
# Real condition is SPAM and the prediction is HAM: False Positive
# Real condition is SPAM and the prediction is SPAM: True Negative
# Real condition is HAM and the prediction is SPAM: False Negative

# We can use confusion matrices to evaluate the machine learning (ML) models.

# Misclassification Rate (Error Rate) = (FP + FN) / (TP + FP + TN + FN)

# Misclassification rate is the metric which shows how often the classifier predicts wrong.

# False positives: Type 1 Error (A doctor tells a men that he is pregnant.)
# False negatives: Type 2 Error (A doctor tells a pregnant women that she is not pregnant.)


# In[128]:


# Scikit-Learn Library

# We can install the scikit-learn library using one of the following commands:
# 1-) pip install scikit-learn
# 2-) conda install scikit-learn

# In this library, model.fit(x_train, y_train) method is used to train the model on the training data.

# We can get the predicted values on the x_test dataset using model.predict(x_test) function.


# In[129]:


import pandas as pd
import numpy as np


# In[130]:


sms_tsv_df = pd.read_csv('smsspamcollection.tsv', sep="\t")
print(sms_tsv_df)
print(type(sms_tsv_df))


# In[131]:


sms_tsv_df.head() # By default, this will bring the first 5 records in the data frame.


# In[132]:


sms_tsv_df.tail() # By default, this will bring the last 5 records in the data frame.


# In[133]:


# The below isnull() call returns a data frame of boolean values and it indicates whether each cell contains 
# a missing value. 
sms_tsv_df.isnull() 


# In[134]:


# The below call will treat the cells with "False" values as 0s and the cells with "True" values as 1s.
# It will return the counts of the missing values for each column/field.
sms_tsv_df.isnull().sum() 


# In[135]:


"""
The Difference(s) Between The CSV Files and TSV Files:

TSV stands for Tab Separated Values. TSV file is a flat file, which uses the Tab character to delimit 
data and reports one time-series per line. CSV stands for Comma Separated Values. CSV file is a flat file, 
which uses the comma (,) character to delimit data and reports one observation per line.

* CSV files use commas to separate values, meaning that each value in the file is separated 
from the next by a comma. 

* TSV files use tabs to separate values, meaning that each value in the file is separated from the next 
by a tab. This format is similar to CSV, but the delimiter character is different, which is tab.
"""


# In[136]:


# How many rows does this sms_tsv_df have ?
row_num = len(sms_tsv_df)
print(row_num)
print("This dataset has "+str(row_num)+" rows.")


# In[137]:


sms_tsv_df['label']


# In[138]:


print(sms_tsv_df['label'])


# In[139]:


print(sms_tsv_df['label'].unique()) # It brings the unique string values from the label column.


# In[140]:


value_counts_spam_ham = sms_tsv_df['label'].value_counts()
print(value_counts_spam_ham)
print()
print()
ham_counts = value_counts_spam_ham['ham']
spam_counts = value_counts_spam_ham['spam']
print(ham_counts)
print(spam_counts)
print("There are "+str(ham_counts)+" emails marked as ham in this dataset.")
print("There are "+str(spam_counts)+" emails marked as spam in this dataset.")


def calculate_spam_ratio(spam_email_count: int, ham_email_count: int):
    if spam_email_count + ham_email_count == 0:
        return 0
    total_email_count = spam_email_count + ham_email_count
    spam_ratio = spam_email_count / total_email_count
    return spam_ratio

def calculate_ham_ratio(spam_email_count: int, ham_email_count: int):
    if spam_email_count + ham_email_count == 0:
        return 0
    total_email_count = spam_email_count + ham_email_count
    ham_ratio = ham_email_count / total_email_count
    return ham_ratio


spam_ratio = calculate_spam_ratio(spam_counts, ham_counts)
ham_ratio = calculate_ham_ratio(spam_counts, ham_counts)

print()
print()

print("The "+str(spam_ratio)+" of the emails are marked as spam.")
print("The "+str(ham_ratio)+" of the emails are marked as ham.")


# In[141]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[142]:


plt.figure(figsize=(12,9))
plt.xscale("log")
bins = 1.15 ** (np.arange(0, 50))
plt.hist(sms_tsv_df[sms_tsv_df['label'] == 'ham']['length'], bins=bins, alpha=0.8, label='ham')
plt.hist(sms_tsv_df[sms_tsv_df['label'] == 'spam']['length'], bins=bins, alpha=0.8, label='spam')
plt.legend(['ham', 'spam']) # It puts the specified classes with different colors inside a box.
plt.xlabel('SMS Length')
plt.ylabel('SMS Count')
plt.title('The Relation Between the Length and Spamness/Hamness of an SMS')
plt.show()

# Note: When we look at the distribution of the below histogram outputs, we can see 
# that the spam text messages are more likely to be longer than the ham text messages.


# * The data on the x-axis represents the message lengths.

# * The data on the y-axis represents the frequency or count of 
# messages with specific lengths falling within the specified bins.


# In[143]:


plt.figure(figsize=(12,9))
plt.xscale("log")
bins = 1.15 ** (np.arange(0, 50))
plt.hist(sms_tsv_df[sms_tsv_df['label'] == 'ham']['punct'], bins=bins, alpha=0.65)
plt.hist(sms_tsv_df[sms_tsv_df['label'] == 'spam']['punct'], bins=bins, alpha=0.65)
plt.legend(['ham', 'spam']) # It puts the specified classes with different colors inside a box.
plt.xlabel('Punctuation Count of an SMS')
plt.ylabel('SMS Count')
plt.title('The Relation Between the Punctuation Count and Spamness/Hamness of an SMS')
plt.show()

# Note: When we look at the distribution of the below histogram outputs, we can see 
# that the spam text messages are more likely to be longer than the ham text messages.


# * The data on the x-axis represents the message lengths.

# * The data on the y-axis represents the frequency or count of 
# messages with specific lengths falling within the specified bins.


# In[144]:


# Building an ML model predicting whether an SMS message is spam or ham based on its length and how many punctuations it contains.

from sklearn.model_selection import train_test_split

# X: This is the feature data.
X = sms_tsv_df[['punct', 'length']]
print(X)

# y: This is the label data.
y = sms_tsv_df['label']
print(y)


# In[145]:


# Random state parameter is specified to get the same split in different runs
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.35, random_state = 10)


# In[146]:


print("The X_train data is: ")
print()
print()
print(X_train)

print()
print()
print("---------------------------------------------------")
print("The X_test data is: ")
print()
print()
print(X_test)

print()
print()
print("---------------------------------------------------")

print("The y_train data is: ")
print()
print()
print(y_train)

print()
print()
print("---------------------------------------------------")

print("The y_test data is: ")
print()
print()
print(y_test)

print()
print()
print("----------------------------------------------------")


# In[147]:


# checking the shapes of the data splits
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[148]:


# import the machine learning model
from sklearn.linear_model import LogisticRegression

# create the instance of the machine learning model with the regularization penalty specified as l2
lr_model = LogisticRegression(penalty = 'l2')


# In[149]:


# train the ML model and make the model fit to the X_train and y_train data.
lr_model.fit(X_train, y_train)


# In[150]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[151]:


predictions = lr_model.predict(X_test)


# In[152]:


print("The predictions of the logistic regression model are: ")
print()
print(predictions)


# In[153]:


# Building the confusion matrix to compare the y_test and predictions
print(confusion_matrix(y_test, predictions))


# In[154]:


df = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['ham','spam'], columns=['ham','spam'])
print(df)


# In[155]:


df


# In[156]:


# displays a classification report
print(classification_report(y_test,predictions))


# In[157]:


# prints the overall accuracy
print(accuracy_score(y_test,predictions))


# In[166]:


############Support Vector Machine Classifier Training ###
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)


# In[167]:


# Naive-Bayes model evaluation starts here.
# displays the confusion matrix to compare the y_test and predictions
print(confusion_matrix(y_test,nb_predictions))


# In[168]:


# displays the classification report
print(classification_report(y_test,nb_predictions))


# In[169]:


# display the accuracy score for the naive-bayes model
print(accuracy_score(y_test, nb_predictions))


# Naive-Bayes model evaluation ends here.


# In[170]:


# Training a Support Vector Machine Classifier Model
from sklearn.svm import SVC
svc_model = SVC(gamma='auto')
svc_model.fit(X_train,y_train)


# In[171]:


svc_predictions = svc_model.predict(X_test)

# Evaluate the support vector machine classifier model with the evaluations metrics of confusion matrix, 
# classification report, and accuracy score
print(confusion_matrix(y_test, svc_predictions))
print()
print(classification_report(y_test, svc_predictions))
print()
print(accuracy_score(y_test, svc_predictions))


# In[172]:





# In[ ]:




