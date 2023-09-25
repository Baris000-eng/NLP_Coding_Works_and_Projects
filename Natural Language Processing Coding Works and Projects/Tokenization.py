#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Tokenization in NLP
# What is Tokenization?
# Tokenization is the process of breaking up the original raw text into component pieces (tokens).
# Tokens are the pieces of the original text.
# Tokens are the basic building blocks of the document object. Everything that helps us to understand
# the meaning of the text is derived from the tokens and their relationships to each other.

# Prefix: Character(s) at the beginning (Examples: ?("$)
# Suffix: Character(s) at the end (Examples: km ).!")
# Infix: Character(s) in between (Examples: /)
# Exception: Special-case rule to split a string into several tokens or prevent a token from being
# split when punctuation rules are applied. (Examples: let's U.S.)


# In[2]:


import spacy
nlp = spacy.load("en_core_web_sm")


# In[8]:


# Escape character of \ is used in the below string. 
# It is to not stop the whole string too early and escape the single quote in the string.
my_string = '"We\'re moving to L.A.!"'
print(my_string)


# In[9]:


# Spacy isolates punctuation that does not form an integral part of a word. One example to this kind of punctuation
# is the question mark at the end of the sentence.
document = nlp(my_string)
for token in document: 
    print(token.text)


# In[11]:


second_string = "We're here to help! Send snap-mail, email livesupport@site.com or visit us at http://www.site.com!"
second_document = nlp(second_string)
for token in second_document:
    print(token.text)


# In[20]:


# applying nlp pipeline of Spacy to the input text and retrieving a document object
third_document = nlp(u"A 5 km NYC Cab ride costs $11.10") 

# iterating through whole document
for token in third_document:
    print(token.text)
    


# In[21]:


doc4 = nlp(u"Let's visit St. Louis in the U.S. next year.")
for token in doc4:
    print(token)


# In[22]:


token_num = len(doc4) # It gets the number of tokens in the document called doc4
print(token_num)


# In[23]:


# Vocab objects contain a full library of items
doc4.vocab


# In[24]:


print(doc4.vocab)


# In[27]:


print(type(doc4.vocab))
print(len(doc4.vocab))


# In[28]:


fifth_document = nlp(u"It is better to give than receive.")
print(fifth_document)
print(fifth_document[0])
print(fifth_document[1:4])


# In[29]:


fifth_document[0] = "test" # It will throw a "spacy.tokens.doc.Doc object does not support item assignment" error.


# In[31]:


sixth_document = nlp(u"Apple Inc. has built a factory to Tokyo for $8 million dollars.")
for token in sixth_document:
    print(token.text, end=" | ")


# In[38]:


# Apple Inc., Tokyo, and $8 million dollars are the named entities 
# detected by Spacy in the given document called 'sixth_document'.

# Displaying each entity and its label in all of the entities in the sixth document.t
for entity in sixth_document.ents:
    print(str(entity) + " => " + entity.label_)


# In[ ]:




