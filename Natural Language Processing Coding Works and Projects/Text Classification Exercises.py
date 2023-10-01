#!/usr/bin/env python
# coding: utf-8

# In[28]:


# necessary imports 
import numpy as np
import pandas as pd

# reading the moviereviews2.tsv file
movie_reviews_df = pd.read_csv("moviereviews2.tsv", sep="\t")
print(movie_reviews_df)


# In[29]:


print(len(movie_reviews_df))
row_num, column_num = movie_reviews_df.shape
print()
print("Row number: "+str(row_num)+", Column number: "+str(column_num)+"")
print("There are "+str(row_num)+" movies in the movie reviews data frame.")


# In[30]:


movie_reviews_df.isnull()


# In[31]:


print(movie_reviews_df.isnull())


# In[32]:


# checks for nan values
movie_reviews_df.isnull().sum()


# In[33]:


# dropping the nan values
movie_reviews_df.dropna(inplace=True)


# In[34]:


movie_reviews_df.isnull().sum()


# In[35]:


# check for whitespace string for both labels and movie reviews

blanks = list()
for index, label, review in movie_reviews_df.itertuples():
    if review.isspace():
        blanks.append(index)
        
print(blanks)
print("There are "+str(len(blanks))+" whitespace strings in the movie reviews data frame.")


# In[36]:


movie_labels = movie_reviews_df['label']
label_counts = movie_labels.value_counts()
print(label_counts)


# In[37]:


print("There are "+str(label_counts['pos'])+" positive movies in the movie reviews data frame.")
print("There are "+str(label_counts['neg'])+" negative movies in the movie reviews data frame.")


# In[38]:


from sklearn.model_selection import train_test_split

X = movie_reviews_df['review']
y = movie_reviews_df['label']

# Dividing the entire dataset into the training set and test set. Allocating 33 percent of the data to the 
# test set and 67 percent of the data to the training set. The 'random_state' parameter is set to get the 
# same result in different splits.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 40)

# displaying the X_train, X_test, y_train and y_test data 
print("-------------------------------------------------")
print("The X_train data is as follows: ")
print()
print(X_train)
print("-------------------------------------------------")
print("The X_test data is as follows: ")
print()
print(X_test)
print("-------------------------------------------------")
print("The y_train data is as follows: ")
print()
print(y_train)
print("-------------------------------------------------")
print("The y_test data is as follows: ")
print()
print(y_test)
print("-------------------------------------------------")


# In[40]:


# Building a pipeline to vectorize the data, and training & fitting a machine learning (ML) model.

from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


text_pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('svc', LinearSVC())])

# fit the data 
text_pipeline.fit(X_train, y_train) 


# In[41]:


print(text_pipeline)


# In[43]:


# Form the predictions 
list_of_predictions = text_pipeline.predict(X_test)


# In[48]:


# Evaluate the performance of the linear support vector classifier model
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report

# displaying the confusion matrix 
print("The confusion matrix is as follows: ")
print()
confusion_matrix = confusion_matrix(y_test, list_of_predictions)
print(confusion_matrix)


print()
print("-----------------------------------------------------------")

# displaying the accuracy score 
print("The overall accuracy score is as follows: ")
print()
accuracy_score = accuracy_score(y_test, list_of_predictions)
print(accuracy_score)

print()
print("-----------------------------------------------------------")

# displaying the classification report 
print("The classification report is as follows: ")
print()
classification_report = classification_report(y_test, list_of_predictions)
print(classification_report)

print()
print("------------------------------------------------------------")


# In[ ]:




