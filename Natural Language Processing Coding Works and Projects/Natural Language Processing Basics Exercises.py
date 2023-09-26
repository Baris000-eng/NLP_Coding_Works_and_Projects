#!/usr/bin/env python
# coding: utf-8

# In[20]:


import spacy 
nlp = spacy.load('en_core_web_sm')
print(nlp)
print(type(nlp))


# In[21]:


with open('owlcreek.txt') as file:
    content = file.read()
    document = nlp(content)
    
print(document)


# In[22]:


span = document[:36]
print(span)


# In[23]:


# How many tokens are contained in the file ?
print(len(document))
print("There are "+str(len(document))+" tokens contained in the file.")


# In[24]:


# How many sentences are contained in the file ?
document_sentences = []
for sentence in document.sents:
    document_sentences.append(sentence)
    
print(document_sentences)
print()
print()
print(len(document_sentences))


# In[25]:


print(len(document_sentences))
print("There are "+str(len(document_sentences))+" sentences in the document.")


# In[26]:


# Grabbing the third sentence in the document
third_sent = document_sentences[2].text
print(third_sent)


# In[28]:


for token in document_sentences[2]:
    print(token, token.text, token.pos_, token.lemma_, token.dep_)


# In[32]:


for token in document_sentences[2]:
    print(f'{token.text:{24}} {token.pos_:{8}} {token.dep_:{16}} {token.lemma_:{24}}')


# In[34]:


# Import the Matcher library:

from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)

# Create a pattern and add it to matcher:
# Write a matcher called 'Swimming' that finds both occurrences of the phrase "swimming vigorously" in the text
pattern = [{'LOWER': 'swimming'}, {'IS_SPACE': True, 'OP':'*'}, {'LOWER': 'vigorously'}]
matcher.add('Swimming', [pattern]) # adding the pattern to the matcher object
matches = matcher(document) # Passing the document to the matcher object to find the matches between the pattern and document.
print(matches)


# In[43]:


def print_surrounding_text(document, start_index, end_index):
    return document[start_index-15:end_index+15]
    

print(print_surrounding_text(document, 1274, 1277))
print(print_surrounding_text(document, 3609, 3612))
print()
print()
print(document[1260:1297]) # printing the surrounding text of the phrase 'swimming vigorously'
print()
print()
print(document[3600:3619]) # printing the surrounding text of the phrase 'swimming vigorously'


# In[92]:


print("---------------------------------------------------------------------------------------------------")
for i in range(0, len(matches)):
    match = matches[i]
    match_id, start, end = match
    for sentence in document_sentences:
        print()
        print("Start: "+str(start)+",", "End: "+str(end)+",", "Sentence Start: "+str(sentence.start)+",", "Sentence End: "+str(sentence.end)+"")
        print()
        if start < sentence.end:
            print("The matched sentence is: "+str(sentence)+"")
            print("---------------------------------------------------------------------------------------------------")
            break


# In[71]:


for match in matches:
    for sentence in document_sentences:
        if match[1] < sentence.end:
            print(sentence)
            break


# In[ ]:




