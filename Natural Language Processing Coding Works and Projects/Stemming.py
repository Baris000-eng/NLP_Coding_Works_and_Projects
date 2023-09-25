#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Often when searching text for a certain keyword, stemming helps if the search returns variations of the word.
# For example, searching for "boat" might also return "boats" and "boating". Here, "boat" would be the stem (root) 
# for [boat, boater, boating, boats]

# Stemming chops off the letters from the end until the stem is reached.

# Spacy library does not include a stemmer. Instead, it prefers to entirely rely on the lemmatization.

# One of the most common and effective stemming tools is Porter's Algorithm. It employs five phases of word 
# reduction, each with its own set of mapping rules. In the first phase, suffix mapping rules are defined.
# More sophisticated phases consider the complexity/length of the word before applying a rule.


# Snowball is the name of a stemming language developed by Martin Porter.
# The algorithm used here is more accurately called the "English Stemmer" or "Porter2 Stemmer".
# It offers a slight improvement over the original Porter Stemmer, both in speed and logic.


# In[7]:


import nltk 
from nltk.stem.porter import PorterStemmer 

porterStemmer = PorterStemmer()
words = ["ran","run","runs","fairly", "easily", "runner", "fairness"]

# show the word itself and its stemmed version for all the words in the list.
for word in words:
    print(word + " ======> " + porterStemmer.stem(word))


# In[8]:


# SnowballStemmer is the improved version of the PorterStemmer
from nltk.stem.snowball import SnowballStemmer

snowballStemmer = SnowballStemmer(language='english')

for word in words:
    print(word + " ======> " + snowballStemmer.stem(word))


# In[9]:


words_lst = ["generate", "generation", "generous", "generously"]


# In[10]:


for word in words_lst:
    print(word + " ======> " + snowballStemmer.stem(word))


# In[ ]:




