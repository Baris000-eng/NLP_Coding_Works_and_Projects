#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Project description: 

# In this project, there is a dataset with which we work. This dataset includes over 400000 quora quetions that 
# have no labeled category. We should try to find 20 categories to assign these questions to.


import pandas as pd

# read the csv file called 'quora_questions.csv'
quora_questions = pd.read_csv('quora_questions.csv')
print(quora_questions)
print()
print()
print(len(quora_questions))
print("There are "+str(len(quora_questions))+" quora questions in this data frame.")


# In[6]:


rows, columns = quora_questions.shape
print("There are "+str(rows)+" rows in the quora questions data frame.")
print("There are "+str(columns)+" columns in the quora questions data frame.")


# In[9]:


# Use TF-IDF Vectorization to create a vectorized document term matrix. You may want to explore 
# max_df and min_df parameters.

# necessary imports 
from sklearn.feature_extraction.text import TfidfVectorizer 

tfidf = TfidfVectorizer(max_df = 0.97, min_df = 2, stop_words = 'english')
document_term_matrix = tfidf.fit_transform(quora_questions['Question'])
row_number, column_number = document_term_matrix.shape


print("The total number of rows in the document term matrix is: "+str(row_number)+"")
print("The total number of columns in the document term matrix is: "+str(column_number)+"")

print("There are "+str(row_number)+" questions in the document term matrix.")
print("There are "+str(column_number)+" words in the document term matrix.")


# In[10]:


# Performing Non-Negative Matrix Factorization (NMF)
from sklearn.decomposition import NMF
nmf_model = NMF(n_components = 11, random_state = 42)
print(nmf_model)
print(type(nmf_model))
print()
print()


# In[11]:


# Fitting the nmf model to the document term matrix 
nmf_model.fit(document_term_matrix)


# In[13]:


# Displaying top 30 most common words for each of the 11 topics

for index, topic in enumerate(nmf_model.components_):
    print("Top 25 words in topic #"+str(index)+": ")
    top30_words = [tfidf.get_feature_names()[j] for j in topic.argsort()[-30:]]
    print(top30_words)
    print()
    


# In[15]:


# Assigning topics to Quora questions using NMF model
topic_outcomes = nmf_model.transform(document_term_matrix)

# Obtaining the list of indices that correspond to the most probable topic for each question
print(topic_outcomes.argmax(axis=1))


# In[16]:


# Creating a new column called 'Topic' and assigning the most probable topic indices to the 
# questions in a rowwise manner.
quora_questions['Topic'] = topic_outcomes.argmax(axis=1)
print(quora_questions)


# In[20]:


print("The head (first 5 records) of the quora questions data frame is: ")
print()
print(quora_questions.head())
print()
print()
print("--------------------------------------------------------------------")
print("The tail (last 5 records) of the quora questions data frame is: ")
print()
print(quora_questions.tail())
print()


# In[ ]:




