#!/usr/bin/env python
# coding: utf-8

# In[1]:


# necessary imports 
import spacy

# loading the small english core language library
nlp = spacy.load('en_core_web_sm') 

print(nlp) 
print(type(nlp))


# In[2]:


document = nlp(u"This is the initial sentence. This is another sentence. This is the last sentence.")
print(document)
print(type(document))
print()
print()

# Displaying each sentence in the document
for sentence in document.sents:
    print(sentence)


# In[3]:


# Displaying each token in the document
for i, token in enumerate(document):
    print("Word "+str(i+1)+": "+str(token)+"")


# In[4]:


# We cannot grab each sentence individually from doc.sents. Because it is a generator object and the generator
# objects are not subscriptable.
document.sents[1]


# In[ ]:


print(type(document.sents)) # gets the type of the generator object of document.sents
print(list(document.sents))
print()
print()

# We should convert the document.sents to a list in order to make it subscriptable.
for i in range(0, len(list(document.sents))):
    print("Sentence "+str(i+1)+": "+str(list(document.sents)[i]))


# In[ ]:


print(type(list(document.sents)[0])) # Span object


# In[ ]:


doc = nlp(u'"Management is doing things right; leadership is doing the right things." - Peter F. Drucker')
print()
print(doc)
print(doc.text)
print(type(doc))


# In[ ]:


for sent in doc.sents:
    print(sent)
    print("\n")


# In[ ]:


# for each token in the document doc, print that token and its index position
for token in doc:
    print(token, token.i)


# In[ ]:


# Ways of adding new rules to the NLP pipeline:

# 1-) Adding a segmentation rule
# 2-) Change segmentation rules


# In[ ]:


from spacy.language import Language

@Language.component("custom_sentence_segmentation_rule")
def custom_sentence_segmentation_rule(doc):
    for token in doc[:-1]:
        if token.text == ";":
            doc[token.i+1].is_sent_start = True
    return doc

# Add the custom sentence segmentation component to the pipeline
#nlp.add_pipe("custom_sentence_segmentation_rule", before="parser")
#nlp.remove_pipe("custom_sentence_segmentation")

print(nlp.pipe_names)
print(type(nlp.pipe_names)) # The list of pipe names is of type SimpleFrozenList.

# SimpleFrozenList is a read-only list-like object in spaCy that is used to store pipeline component names. 


# In[ ]:


print(doc[:-1]) # Display all of the tokens up to but not including the last one.


# In[ ]:


doc5 = nlp(u'"Management is doing things right; leadership is doing the right things." - Peter F. Drucker')
print(doc5)
print(type(doc5))
print()
print()

# As we can see from the output of this for loop, after adding a custom sentence segmentation rule of segmenting 
# the sentences based on the semicolon, the sentences in the output are splitted based on the semicolon.
for sent in doc5.sents:
    print(sent)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


# ALTER SENTENCE SEGMENTATION RULES
nlp = spacy.load('en_core_web_sm') # reloading the english core language library
myStr = u"This is the first sentence.\nThis is the second sentence.\n\nThis is the third sentence.\nThis is the \nfourth sentence."
print(myStr)


# In[21]:


custom_document = nlp(myStr)
print(custom_document)
print()
print(type(custom_document)) # The document called 'custom_document' is a Doc object. It has a class of spacy.tokens.doc.Doc.


# In[22]:


for sentence in custom_document.sents:
    print(sentence)


# In[23]:


# necessary imports
from spacy.language import Language

print(nlp.pipe_names)
@Language.component("split_on_newlines") 
def split_on_newlines(doc):
    start = 0
    newLineEncountered = False
    for word in doc:
        if newLineEncountered:
            start = word.i
            newLineEncountered = False
        elif word.text.startswith("\n"): # a new line has been encountered.
            newLineEncountered = True
            
        
    return doc[start:]

nlp.add_pipe("split_on_newlines")
print(type(myStr))
custom_doc = nlp(myStr)
print(custom_doc)
print(nlp.pipe_names)


# In[24]:


split_on_newlines(custom_doc)


# In[25]:


for sent in custom_doc.sents:
    print(sent)


# In[ ]:





# In[ ]:





# In[ ]:




