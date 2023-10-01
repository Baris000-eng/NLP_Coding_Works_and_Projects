#!/usr/bin/env python
# coding: utf-8

# In[5]:


import spacy
from spacy.matcher import PhraseMatcher
nlp = spacy.load('en_core_web_sm')

# Adding multiple phrases as NER parts
new_doc = nlp(u"Our company created a brand new washing machine."
              u" This new washing-machine is the best in show.")
print(new_doc)
print(type(new_doc))

phraseMatcher = PhraseMatcher(nlp.vocab)
print(phraseMatcher)
print(type(phraseMatcher))


# In[6]:


list_of_phrases = ["washing-machine", "washing machine"]
phrase_patterns = []
for phrase in list_of_phrases:
    doc = nlp(phrase)
    phrase_patterns.append(doc)
    
print(str(phrase_patterns) + " ===> " + str(type(phrase_patterns)))
phraseMatcher.add('WashingMachine', phrase_patterns)
found_matches = phraseMatcher(new_doc) # passing the new_doc document to the phrase matcher to find a sequence of matches with the document
print(found_matches)
print(new_doc[8], new_doc[9], new_doc[10], new_doc[11])


# In[13]:


from spacy.tokens import Span
prod = doc.vocab.strings[u"PROD"]
print(prod) # hashcode of PRODUCT named entity
new_entities = []
for match in found_matches:
    span = Span(new_doc, match[1], match[2], label=prod) # gets the span of the match
    new_entities.append(span)
    
print(new_entities)
print(type(new_entities))

print()
print()

new_doc.ents = list(new_doc.ents) + [new_entities]


# In[14]:


def display_entities(document):
    if len(document.ents) == 0:
        raise Exception("No entities are found!")
    else:
        for entity in document.ents:
            print(str(entity) + " ==> " + entity.text + " ==> " + str(entity.label_) + " ==> " + str(spacy.explain(entity.label_)))
display_entities(new_doc)


# In[ ]:




