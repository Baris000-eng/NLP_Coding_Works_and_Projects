#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Topic Modelling #

# Topic Modelling Overview 

# Topic Modelling allows for us to efficiently analyze large volumes of text by clustering 
# the documents into topics.

# A large amount of text data is unlabeled, which means we will not be able to apply the 
# supervised learning approaches to create machine learning models for the data. Because 
# the supervised machine learning approaches depend on the historical labelled data.

# It is up to use to try to discover text labels through the usage of topic modelling.

# If we have unlabeled data, then we can attempt to "discover" the labels. In the case of 
# text data, this means attempting to discover clusters of similar documents, grouped 
# together by topic.

# A very important idea to keep in mind here is that we do not know the "correct topic" or
# the "right answer". All we know is that the documents which are clustered together share
# similar topic ideas. It is up to the user to determine what these topics represent.


# In[2]:


# Latent Dirichlet Allocation (LDA) for Topic Modelling

# * Johann Peter Gustav Lejeune Dirichlet was a German mathematician in the 1800s who 
# contributed widely to the field of modern mathematics.

# There is a probability distribution named after him, "Dirichlet Distribution". The 
# Latent Dirichlet Allocation (LDA) is based on this probability distribution.

# In 2003, LDA was first published as a graphical model for topic discovery in 
# Journal of Machine Learning research.

# Assumptions of LDA for Topic Modelling

# 1-) Documents with similar topics use similar groups of words
# 2-) Latent topics can be found by searching for groups of words which frequently occur together 
# in documents across the corpus.

# - Documents are probability distributions over latent topics.
# - Topics themselves are probability distributions over words.

# LDA represents documents as mixtures of topics which generate words with certain probabilities.


# LDA assumes that the documents are produced in the following fashion: 

# - Decide on the number of words N the document will have.
# - Choose a topic mixture for the document (according to the Dirichlet distribution over a fixed set of K topics)
   # * e.g. 55% business, 25% politics, 10% economics 10% trade
    
# Generate each word in the document by: 
  # First picking a topic according to the multinomial distribution that is sampled in the previous step 
  # (55% business, 25% politics, 10% economics, and 10% trade).

# Use the topic to generate the word itself (according to the topic's multinomial distribution). For instance,
# if we choose the topic named 'economics', we might generate the word 'stocks' with 50% probability, 
# 'investment' with 35% probability, and so forth.

# Assuming this generative model for a collection of documents, LDA then tries to backtrack from the documents 
# to find a set of topics that are likely to have generated the collection.


# We have choosen a fixed number of K topics, and want to use LDA to learn the topic representation of each 
# document and the words associated to each topic.

# We are going to go through each document, and randomly assign each word in the document to one of the K topics.

# This random assignment gives us the topic representations of all the documents and word distributions of all 
# the topics.

# We iterate through every word in every document to improve this fixed set of topics.

# For every word in every document and for each topic t, we calculate the following: 
# p(topic t | document d) = the proportion of words in the document d that are currently assigned to topic t.

# Reassign w a new topic, where we choose topic t with the below probability: 

# p(topic t | document d) * p(word w | topic t)
# This probability is essentially the probability that the topic t generated word w.

# After repeating the previous step for many times, we finally reach an approximately steady state where 
# the assignments are acceptable.

# Ultimately, we have each document being assigned to a topic. We can also search for the words which have 
# the highest probability of being assigned to a topic.

# We end up with an output such as: 

# Document assigned to the Topic #4
# Most common words (highest probability) for Topic #4:
# ['cat', 'dog', 'vet', 'birds', 'food', 'home', ...]

# Two important notes: 
# 1-) The user must decide on the amount of topics present in the document.
# 2-) The user must interpret what the topics are.


# In[39]:


# Latent Dirichlet Allocation
import pandas as pd
npr = pd.read_csv('npr.csv')
print(npr)
print()
print(type(npr))


# In[40]:


print(npr['Article'])


# In[41]:


print("There are "+str(len(npr['Article']))+" articles in this dataset.")


# In[42]:


# data preprocessing
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(min_df=0.15, max_df=0.95, stop_words = 'english')
print(cv)
print(type(cv))


# In[43]:


document_term_matrix = cv.fit_transform(npr['Article'])
print(document_term_matrix)


# In[44]:


document_term_matrix


# In[45]:


# Apply Latent Dirichlet Allocation
from sklearn.decomposition import LatentDirichletAllocation

# creating an lda instance where the random state is equal to 42 and the number of topics seeked is 20
lda = LatentDirichletAllocation(n_components=20, random_state=42)


print(lda)
print(type(lda))


# In[46]:


# fitting lda to the document_term_matrix
lda.fit(document_term_matrix)


# In[47]:


# grab the vocabulary of words 

words_vocab = cv.get_feature_names()
print(len(words_vocab))
print()
print(type(words_vocab))
print()
for i in range(0, 100):
    print(words_vocab[i])


# In[62]:


print(type(words_vocab))

import random 

random_word_id = random.randint(0, 185)
print(random_word_id)


# In[ ]:





# In[ ]:





# In[64]:


# grab the topics
topics = lda.components_
print(topics)
print()
print(len(topics))
print("The type of the topics is: "+str(type(topics))+"")
print("The type of the lda.components_ is: "+str(type(lda.components_))+"")

topics_shape = topics.shape
print(topics_shape)

row_num = topics.shape[0]
col_num = topics.shape[1]

print("Total number of rows: "+str(row_num)+"")
print("Total number of columns: "+str(col_num)+"")
print()
print("There are "+str(row_num)+" topics seeked and "+str(col_num)+" words in the topics data.")


# In[53]:


single_topic = lda.components_[0]
print("The first topic: "+str(single_topic)+"")


# In[56]:


single_topic.argsort()


# In[66]:


# It gets the list of index positions of the high probability words for the first topic, 
# meaning the topic located at the 0th index.
print(single_topic.argsort())


# In[72]:


print(single_topic.argsort()[-10:]) # top 10 high-probable words


# In[80]:


# Displaying top 20 high-probable words that can show up in the topic called 'single_topic'

top_twenty_words = single_topic.argsort()[-20:]
feature_names = cv.get_feature_names()
for index in top_twenty_words:
    print(feature_names[index])
    


# In[81]:


import numpy as np
arr = np.array([20, 300, 11])
print(arr)


# In[82]:


print(arr)

# sorts the values in an ascending order and returns the indices of these values in a sequence
# Since 11 is the smallest value in the numpy array called 'arr', its index (2) comes first in the 
# argsort() call. 20 is greater than 11 and it is the smallest choice that we can take which is
# larger than 11. So, the index of '20' (0) takes the second place in the output. Because of the 
# fact that 300 is the largest value in the numpy array called arr, its index (1) takes the last 
# index position in the output of argsort() call.

# In short, argsort() returns the index positions that will sort the array with which argsort() is called.
print(arr.argsort()) 


# In[109]:


# grab the highest probability words per topic
for index, topic in enumerate(topics):
    print("The top 15 words for the topic "+str(index)+" are: ")
    print()
    top15_words = [cv.get_feature_names()[index] for index in topic.argsort()[-15:]]
    print(top15_words)
    print()
    print("------------------------------------------------------------------------")


# In[87]:


document_term_matrix


# In[88]:


npr


# In[99]:


topic_results = lda.transform(document_term_matrix)
print(topic_results)
print(type(topic_results))
print("\n")
print("\n")
print(topic_results.shape[0], topic_results.shape[1])
print("There are "+str(topic_results.shape[0])+" articles presented and "+str(topic_results.shape[1])+" topics seeked in the data frame.")


# In[97]:


# probabilities of the first article belonging to a particular topic for each topic in the list of topics
print(topic_results[0])


# In[98]:


# The below call will round every element in the first element of the array called topic
topic_results[0].round(4)


# In[101]:


topic_results[0].round(3)


# In[102]:


topic_results[0].argmax() # It returns the index position of the highest probability topic for the first article.


# In[105]:


print(topic_results)
print()
print()
print()
print()
npr['Topic'] = topic_results.argmax(axis=1)
print(npr)


# In[104]:


print(npr['Article'][0])


# In[108]:


print(topic_results)
print()
print()
print(topic_results.argmax(axis=0)) # columnwise list of indices each of which correspond to the maxium value in the column
print(topic_results.argmax(axis=1)) # rowwise list of indices each of which correspond to the maximum value in the row


# axis = 0 ======> columnwise
# axis = 1 ======> rowwise


# In[ ]:




