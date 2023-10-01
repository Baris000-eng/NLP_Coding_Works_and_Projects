#!/usr/bin/env python
# coding: utf-8

# In[18]:


# Non-Negative Matrix Factorization

# Non-Negative Matrix factorization is an unsupervised learning algorithm that simultaneously 
# performs dimensionality reduction and clustering. 

# We can use this technique in conjunction with TF-IDF to model topics across documents.

# We are given a non-negative matrix A, find k-dimension approximation in terms of the 
# non-negative factors W and H.


# Basis Vectors ==> W 
# Coefficient Matrix ==> H

# n * m (data matrix, rows = features, cols = objects) ===> n * k (W, Basis Vectors, rows = features)  k * m (H, Coefficient Matrix, cols = objects)
# Note: W >= 0 and H >= 0

# Approximate each object (column of A) by a linear combination of k reduced dimensions or "basis vectors" in W.

# Each basis vector can be interpreted as a cluster. The memberships of objects in these clusters are encoded by H.


# Input: Non-negative data matrix (A), number of basis vectors (k), and initial values for 
# the factors W and H (e.g. random matrices). In topic modelling, The 'k' is the number of 
# topics we choose. Here, the 'A' is the TF-IDF for the words across all the documents.

# Objective function: Some measure of reconstruction error between A and the approximation WH.

# Expectation-maximization optimization to refine W and H in order to minimize the objective function.
# Common approach is to iterate between two multiplicative update rules until convergence.

# Steps: 

# 1-) Construct a vector space model for the documents (after stopword filtering), which results in a 
# term document matrix A.

# 2-) Apply TF-IDF term weight normalization to A.

# 3-) Normalize TF-IDF vectors to unit length.

# 4-) Initialize the factors using non-negative double singular value decomposition (NNDSVD) on A.

# 5-) Apply a projected gradient non-negative matrix factorization to A.

# * Basis Vectors: The topics (clusters) in the data.
# * Coefficient Matrix: The membership weights for documents relative to each topic (cluster).


#---------------------------------------------------------------------------------------------------#

# * Create a document term matrix with TF-IDF vectorization.
# * Resulting W and H.
# Basis vectors W = Topics (Clusters)
# Coefficients H = Memberships for documents


# Important Notes: 
# Just like LDA, we will need to select the number of expected topics beforehand (the value of k)!
# Moreover, just like LDA, we will have to interpret the topics based on the coefficient values
# of the words per topic.
# Coefficient value is not a probability value like the LDA gives us.


"""Comparison between Latent Dirichlet allocation (LDA) and Non-negative Matrix Factorization (NMF) –

Latent Dirichlet allocation (LDA)

* Assumes each document has multiple topics.
* Works best with longer texts such as full articles, essays, and books.
* Evolves as you process new documents with the same model.
* Results are not deterministic, meaning you might get different results each time for the same data set.

Non-negative Matrix Factorization(NMF)

* Calculates how well each document fits each topic, rather than assuming a document has multiple topics.
* Usually faster than LDA.
* Works best with shorter texts such as tweets or titles.
* Results are almost deterministic, having more consistency when running the same data."""


# In[19]:


import pandas as pd

npr = pd.read_csv('npr.csv')
print(npr)
print()
print(type(npr))


# In[20]:


npr.head()


# In[21]:


npr.tail()


# In[22]:


print(npr.head())
print()
print()
print(npr.tail())


# In[23]:


from sklearn.feature_extraction.text import TfidfVectorizer 

tfidf = TfidfVectorizer(max_df = 0.90, min_df = 2, stop_words = 'english')
print(tfidf)
print(type(tfidf))


# In[24]:


document_term_matrix = tfidf.fit_transform(npr['Article'])
print(document_term_matrix)
print(type(document_term_matrix))


# In[25]:


document_term_matrix


# In[26]:


# Perform Non-Negative Matrix Factorization 
from sklearn.decomposition import NMF


nmf_model = NMF(n_components = 10, random_state = 42)
nmf_model.fit(document_term_matrix)


# In[30]:


# As you can see below, the length of the 'feature_names' array is 54777. This means that there are 54777 unique 
# terms or words in the corpus or collection of documents which are used to create the TF-IDF representation. 

feature_names = tfidf.get_feature_names()
print(len(feature_names))

print(f"There are {len(feature_names)} unique terms or words in the corpus used to create the TF-IDF representation.")

print()
print()
print()
print()
for j in range(0, 1000):
    print(feature_names[j])


# In[33]:


for index, topic in enumerate(nmf_model.components_):
    print(f"The top 25 words for topic#{index}: ")
    print()
    top25_words = [tfidf.get_feature_names()[j] for j in topic.argsort()[-25:]]
    print(top25_words)
    print()


# In[41]:


# It attaches discovered topic labels to the original articles
topic_results = nmf_model.transform(dtm)
print(topic_results.argmax(axis=1))


# assigning the index of the maximum value for each topic to the numerical topic label in the npr data frame.
# creating a numerical topic label column called 'Topic' in the npr data frame.
npr['Topic'] = topic_results.argmax(axis=1) 


# In[42]:


npr.head()


# In[43]:


npr.tail()


# In[44]:


print(npr.head())


# In[45]:


print(npr.tail())


# In[ ]:




