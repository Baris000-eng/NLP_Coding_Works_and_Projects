#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

df = pd.read_csv('moviereviews.tsv', sep='\t')
print(df)


# In[5]:


row_num, col_num = df.shape
print()
print("The total number of rows in this movie reviews data frame: "+str(row_num)+"")
print("The total number of columns in this movie reviews data frame: "+str(col_num)+"")
print()


# In[6]:


df.isnull()


# In[7]:


df.isnull().sum()


# In[23]:


# As we can see in the output, there are 35 reviews missing in the 'review' column.
print(df.isnull().sum())


# In[24]:


df.dropna(inplace=True)
print(df.isnull().sum())


# In[25]:


blanks = list()
for index, label, review in df.itertuples():
    if type(review) == str and review.isspace():
        blanks.append(index)
        
print(blanks)


# In[26]:


df.drop(blanks, inplace = True)


# In[27]:


print(df.isnull().sum())


# In[22]:


# We can either use df.drop(blank_positions, inplace=True) or df.dropna(inplace=True) method to drop the null 
# values from the data frame called 'df'.


# In[28]:


df['label'].value_counts()


# In[29]:


# gets the total number of each label type in the 'label' column of the dataframe called 'df'
labels = df['label']
print(labels.value_counts())


# In[31]:


neg_count = labels.value_counts()['neg']
pos_count = labels.value_counts()['pos']
print("The total number of negative reviews in the movie reviews data frame: "+str(neg_count)+"")
print("The total number of positive reviews in the movie reviews data frame: "+str(pos_count)+"")


# In[34]:


# Perform sentiment analysis 

# Necessary imports
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Creating a 'SentimentIntensityAnalyzer' object
sia = SentimentIntensityAnalyzer()

df['scores'] = df['review'].apply(lambda review: sia.polarity_scores(review))
df['compound'] = df['scores'].apply(lambda score: score['compound'])

print(df)


# In[37]:


df['predicted_sentiment_label'] = df['compound'].apply(lambda score: "neg" if score < 0 else "pos")


# In[38]:


print(df.head())


# In[45]:


# Neccessary imports for the evaluation metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[46]:


# Performance evaluation on sentiment prediction

true_values = df['label']
predicted_values = df['predicted_sentiment_label']

accuracy = accuracy_score(true_values, predicted_values)
confusion_matrix = confusion_matrix(true_values, predicted_values)
classification_report = classification_report(true_values, predicted_values)

print("The overall accuracy is as below: ")
print()
print(accuracy)

print()
print("---------------------------------------------------------")

print("The confusion matrix is as below: ")
print()
print(confusion_matrix)

print()
print("---------------------------------------------------------")

print("The classification report is as below: ")
print()
print(classification_report)

print()
print("---------------------------------------------------------")


# In[ ]:




