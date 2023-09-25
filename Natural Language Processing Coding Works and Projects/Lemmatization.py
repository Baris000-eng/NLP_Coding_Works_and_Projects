#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Lemmatization is a text pre-processing technique used in natural language processing (NLP) models 
# to break a word down to its root meaning to identify similarities. For example, a lemmatization 
# algorithm would reduce the word better to its root word, or lemme, good.  

# In contrast to stemming, lemmatization looks beyond word reduction and considers a language's full vocabulary 
# to apply a morphological analysis to words.

# The lemma of 'was' is 'be' and the lemma of 'mice' is 'mouse'.

# Lemmatization is much more informative than stemming, which is why the Spacy library has opted to only have 
# lemmatization available instead of stemming. 

# Lemmatization looks at surrounding text to determine a given word's part of speech.


# In[5]:


import spacy 

nlp = spacy.load('en_core_web_sm')

first_document = nlp(u"I am a runner running in a race because I love to run since I ran today.")
for token in first_document:
    print(token, '\t', token.text, '\t', token.pos_, '\t', token.lemma, '\t', token.lemma_)
    
# token.text gives the text content of the token.
# token.pos_ gives the part of speech for the token (Examples: verb, adjective, adverb, noun, ...)
# token.lemma gives a number that points a specific lemma inside the loaded language library.
# Each of the word in the language model has an individual hash to its lemma which we can reference.
# token.lemma_ refers to the actual lemma of the token (Example: The lemma of 'am' is 'be').


# In[6]:


def display_lemmas(text: str):
    for token in text:
        print(f'{token.text:{12}} {token.pos_:{6}} {token.lemma:<{22}} {token.lemma_}')


display_lemmas(first_document)


# In[7]:


second_document = nlp(u'I saw 10 mice today.')
display_lemmas(second_document)


# In[ ]:




