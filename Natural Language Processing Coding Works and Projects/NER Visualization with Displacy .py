#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Visualization of the Named Entity Recognition (Visualization of NER)
import spacy
nlp = spacy.load('en_core_web_sm')
print(nlp)
print(type(nlp))


# In[2]:


from spacy import displacy
document = nlp(u"Over the last three years, Apple sold nearly 200000 Iphones for a profit of 100 million dollars."
               u"On the contrary, Sony only sold 5 thousand Walkman music players.")


# In[3]:


print(document)
print(type(document))
print(document.ents)
print()


# In[4]:


# In the below function call, style is specified as entity through the assignment of 'ent' to the style argument.
displacy.render(document, style='ent', jupyter=True)


# In[5]:


# Compared to the above call, the difference is that this makes each sentence in the docuent left-aligned.
for sentence in document.sents:
    doc_text = nlp(sentence.text)
    displacy.render(doc_text, style='ent', jupyter=True)


# In[6]:


options = {'ents': ['ORG']}

# Since the options dictionary specifies only the organization named entity (ORG), the below function call will
# highlight the organizations (ORG) as the named entities.
displacy.render(document, style='ent', jupyter=True, options=options)


# In[7]:


# We can also specify multiple named entities in the options dictionary as below:

options_multiple = {'ents':['ORG', 'PRODUCT', 'CARDINAL']}
displacy.render(document, style='ent', jupyter=True, options=options_multiple)


# In[8]:


from spacy.matcher import PhraseMatcher
phraseMatcher = PhraseMatcher(nlp.vocab)
print(phraseMatcher)
print(type(phraseMatcher))

list_of_phrases = ["Walkman music players", "Iphones"]
phrase_patterns = []
for phrase in list_of_phrases:
    doc = nlp(phrase)
    phrase_patterns.append(doc)
    
print(str(phrase_patterns) + " ===> " + str(type(phrase_patterns)))
phraseMatcher.add('Products', phrase_patterns)
found_matches = phraseMatcher(document) # passing the document to the phrase matcher to find a sequence of matches with the document
print(found_matches)


# In[9]:


from spacy.tokens import Span
prod = doc.vocab.strings[u"PRODUCT"]

new_entities = []
for match in found_matches:
    span = Span(document, match[1], match[2], label=prod) # gets the span of the match
    new_entities.append(span)
    
print(new_entities)
print(type(new_entities))

print()
print()

doc_ents_lst = list(document.ents)
print(doc_ents_lst)
print("Before appending the new entities: "+str(doc_ents_lst)+"")
doc_ents_lst.append(new_entities)
print(doc_ents_lst)
print("After appending the new entities: "+str(doc_ents_lst)+"")


# In[10]:


print(document.ents)


# In[11]:


colors = {'ORG': 'radial-gradient(yellow, blue)'}
options = {'ents': ['ORG','CARDINAL'], 'colors': colors}
displacy.render(document, style='ent', options=options)


# In[12]:


colors_dict = {'ORG': 'linear-gradient(45deg, red, orange)'}
options_dict = {'ents': ['PRODUCT', 'ORG'], 'colors': colors_dict}
displacy.render(document, style='ent', options=options_dict)


# In[ ]:


displacy.serve(document, style='ent', options=options, port=8003)


# In[ ]:





# In[ ]:




