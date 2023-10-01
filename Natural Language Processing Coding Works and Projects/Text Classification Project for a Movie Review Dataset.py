#!/usr/bin/env python
# coding: utf-8

# In[63]:


# necessary imports 
import numpy as np
import pandas as pd


# In[64]:


# The seperator is tab since the file being read is a tsv (tab seperated values) file.
movie_reviews_df = pd.read_csv("moviereviews.tsv", sep = "\t")
print(movie_reviews_df)


# In[65]:


# head of the movie_reviews data frame (i.e. The first 5 records)
movie_reviews_df.head()


# In[66]:


# tail of the movie_reviews data frame (i.e. The last 5 records)
movie_reviews_df.tail()


# In[67]:


print("The shape of the movie_reviews data frame is: "+str(movie_reviews.shape)+"")


# In[68]:


print(len(movie_reviews_df))
print(movie_reviews_df.shape[0])
print(movie_reviews_df.shape[1])
print("There are "+str(len(movie_reviews))+" movies in the movie_reviews data frame.")


# In[69]:


print("The movie review labels are as follows: ")
print()
movie_reviews = movie_reviews_df['review']
movie_review_labels = movie_reviews_df['label']
for i in range(0, len(movie_review_labels)):
    print(movie_review_labels[i])


# In[70]:


first_movie_review = movie_reviews[0]
first_movie_label = movie_review_labels[0]
print(first_movie_label)
print()
print(first_movie_review)


# In[71]:


second_movie_review = movie_reviews[1]
second_movie_label = movie_review_labels[1]
print(second_movie_label)
print()
print(second_movie_review)


# In[72]:


third_movie_review = movie_reviews[2]
third_movie_label = movie_review_labels[2]
print(third_movie_label)
print()
print(third_movie_review)


# In[73]:


# checks the missing values
movie_reviews_df.isnull().sum()


# As we can see in the below output, we are missing 35 movie reviews in the data frame.


# In[74]:


# permanently dropping the null values from the movie_reviews data frame
movie_reviews_df.dropna(inplace=True)


# In[75]:


movie_reviews_df.isnull().sum()


# In[76]:


blanks = []
for index, label, review in movie_reviews_df.itertuples():
    if review.isspace():
        blanks.append(index)
        
        
# This will display the indexes of the reviews which consists of a space or multiple spaces.
print(blanks)


# In[77]:


# dropping the empty string data 
movie_reviews_df.drop(blanks, inplace=True)


# In[78]:


print(len(movie_reviews_df))


# In[79]:


from sklearn.model_selection import train_test_split 


X = movie_reviews_df['review']
y = movie_reviews_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 39)


# In[80]:


# Displaying the X_train, X_test, y_train, and y_test data 

print("-------------------------------------------------")
print("The X_train data is as below: ")
print()
print(X_train)
print("------------------------------------------------")
print("The X_test data is as below: ")
print()
print(X_test)
print("-------------------------------------------------")
print("The y_train data is as below: ")
print()
print(y_train)
print("-------------------------------------------------")
print("The y_test data is as below: ")
print()
print(y_test)
print("--------------------------------------------------")


# In[81]:


# Building a pipeline to vectorize the movie reviews data 

from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC 

text_pipeline = Pipeline([('tfidf', TfidfVectorizer()),
                          ('svc', LinearSVC())])


# In[82]:


text_pipeline.fit(X_train, y_train) # fits the text pipeline object to the training data


# In[83]:


# Generating the predictions on the X_test data
predictions = text_pipeline.predict(X_test)


# In[86]:


# Evaluation of the Linear Support Vector Classifier model by using several evaluation 
# metrics including the confusion matrix, the accuracy score, and the classification report

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score

print("The confusion matrix is as follows: ")
print()
print(confusion_matrix(y_test, predictions)) # confusion matrix


print()
print("----------------------------------------------------")
print()

print("The classification report is as follows: ")
print()
print(classification_report(y_test, predictions)) # classification report

print()
print("----------------------------------------------------")
print()

print("The overall accuracy score is as follows: ")
print()
print(accuracy_score(y_test, predictions)) # accuracy score
print()
print("----------------------------------------------------")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




