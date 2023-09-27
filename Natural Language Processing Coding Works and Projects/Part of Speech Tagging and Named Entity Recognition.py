#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Part of Speech Tagging

# Part of Speech (POS) Types:
# * Coarse Part of Speech (token.pos_ gets the coarse part of speech )
# * Fine-Grained Part of Speech
import spacy

nlp = spacy.load('en_core_web_sm')
print(nlp)
print(type(nlp))

print()
print()

document = nlp(u"The quick black wolf jumped over the lazy cat's back.")
print(document, "Type: "+str(type(document))) # Doc object in spacy
print(document.text, "Type: "+str(type(document.text))) # String

print()
print()

for i in range(0, len(document)):
    word = document[i]
    print(type(word)) # Here, each word in the document has a type of spacy token.
    print("Word "+str(i+1)+": "+str(word)+"") 
    print()
    


# In[2]:


# Obtaining the texts of the tokens in the document called 'document'.
texts_of_tokens = []
for token in document:
    text = token.text
    texts_of_tokens.append(text)
    
print(texts_of_tokens)
print()
print(texts_of_tokens[0])
print(texts_of_tokens[1])
print(texts_of_tokens[2])
print(texts_of_tokens[3])


# In[3]:


# Obtaining the coarse (general/broad) part of speech tag for each token in the document called 'document'
coarse_part_of_speech_lst = []
for token in document:
    pos = token.pos_ # gets the coarse part of speech for each token
    coarse_part_of_speech_lst.append(""+str(token)+ " => "+str(pos)+"")

print("The list of coarse part of speech tags for the tokens in the document: ")
print()
print(coarse_part_of_speech_lst)
print()
print(coarse_part_of_speech_lst[0])
print(coarse_part_of_speech_lst[1])
print(coarse_part_of_speech_lst[2])
print(coarse_part_of_speech_lst[3])
print(coarse_part_of_speech_lst[4])


# In[4]:


# Obtaining the explanation of the coarse part of speech tag for each token in the document called 'document'
coarse_tags_explanations = []
for word in document:
    pos = word.pos_ # gets the coarse part of speech for each token
    exp_text = spacy.explain(pos) # gets the detailed explanation of the coarse part of speech
    coarse_tags_explanations.append(""+str(word)+" => "+str(exp_text))
    
print("The list of explanations of the coarse part of speech tags for the tokens in the document: ")
print()
print(coarse_tags_explanations)
print()
print(coarse_tags_explanations[0])
print(coarse_tags_explanations[1])
print(coarse_tags_explanations[2])
print(coarse_tags_explanations[3])
print(coarse_tags_explanations[4])
print(coarse_tags_explanations[5])


# In[5]:


# Obtaining the numerical id of the coarse part of speech tag for each token in the document called document
coarse_tags_numerical_ids = []
for token in document:
    coarse_pos_id = token.pos
    coarse_tags_numerical_ids.append(""+str(token)+" => "+str(coarse_pos_id))
    
print("The list of numerical ids of the coarse part of speech tags for the tokens in the document: ")
print()
print(coarse_tags_numerical_ids)
print()
print(coarse_tags_numerical_ids[0])
print(coarse_tags_numerical_ids[1])
print(coarse_tags_numerical_ids[2])
print(coarse_tags_numerical_ids[3])


# In[6]:


# Obtaining the fine-grained part of speech tag for each token in the document called document
fine_grained_pos_lst = []
for token in document:
    tag = token.tag_ # gets the fine-grained POS tag for each token
    fine_grained_pos_lst.append(""+str(token)+" => "+str(tag)+"")
    
print("The list of fine-grained part of speech tags for the tokens in the document: ")
print()
print(fine_grained_pos_lst)
print()
print(fine_grained_pos_lst[0]) # first element of the list 
print(fine_grained_pos_lst[1]) # second element
print(fine_grained_pos_lst[2]) # third element
print(fine_grained_pos_lst[3]) # fourth element


# In[7]:


# Obtaining the explanation of the fine-grained part of speech tag for each token in the document called document
explanations = []
for token in document:
    tag = token.tag_
    explanation = spacy.explain(tag) # gets the detailed explanation of the fine-grained POS tag.
    explanations.append(""+str(token)+" => "+str(explanation)+"")
    
print("The list of the explanations of the fine-grained part of speech tags for the tokens in the document:")
print()
print(explanations)
print()
print(explanations[0]) # The first element of the list
print(explanations[1]) # The second element
print(explanations[2]) # The third element
print(explanations[3]) # The fourth element


# In[8]:


# Obtaining the numerical id of the fine-grained part of speech tag for each token in the document called document
fine_grained_tag_ids = []
for token in document:
    tag_id_value = token.tag # gets the numerical id of the fine-grained POS tag for each token
    fine_grained_tag_ids.append(""+str(token)+" => "+str(tag_id_value)+"")
    
print("The list of fine-grained part of speech tags for the tokens in the document:")
print()
print(fine_grained_tag_ids)
print()
print(fine_grained_tag_ids[0]) # The first element of the list
print(fine_grained_tag_ids[1]) # The second element
print(fine_grained_tag_ids[2]) # The third element
print(fine_grained_tag_ids[3]) # The fourth element
print(fine_grained_tag_ids[4]) # The fifth element


# In[9]:


new_document = nlp(u"I am currently studying NLP techniques such as lemmatization, tokenization, stemming, stop words removal, and so on.")
print(new_document)
print(new_document.text) # gets the text in the document
print(new_document.sents) # gets the sentences in the document
print(type(new_document)) # gets the type of the document object


# In[10]:


# This will output a dictionary where the keys are the part of speech codes and the values are the frequencies 
# of the the parts of speech in the document called 'new_document'.
print(spacy.attrs.POS)
pos_counts = new_document.count_by(spacy.attrs.POS)
print(pos_counts)


# In[11]:


pos_codes = pos_counts.keys()
frequencies = pos_counts.values()
pos_codes = list(pos_codes)
frequencies = list(frequencies)

pos_dict = {} # Initializing a dictionary to keep track of the 
for i, pos_code in enumerate(pos_codes):
    pos_text = nlp.vocab[pos_code].text # gets the text version of the part of speech code
    print(""+str(pos_text)+" => "+str(pos_code)+"")
    pos_dict[pos_text] = frequencies[i]
    
print()
print(pos_dict)


# In[12]:


# Visualizing parts of speech by using displaCy
import spacy 
nlp = spacy.load('en_core_web_sm')
document = nlp(u"I have three apples and 1 kilogram strawberries in my grocery bag.")

from spacy import displacy 
options = {'color': 'blue', 'bg': 'orange', 'font': 'Times New Roman'}
displacy.render(document, style='dep', jupyter=True, options=options)


# In[ ]:


displacy.serve(document, style='dep', port=8888)


# In[ ]:


# Named entity recognition
# Named entity recognition (NER) seeks to locate and classify named entity mentions in 
# the unstructured text into pre-defined categories such as the locations, organizations,
# person names, time expressions, medical codes, quantities, and percentages.

import spacy
nlp = spacy.load('en_core_web_sm')


# In[ ]:


print(nlp)
print(type(nlp))


# In[ ]:


def display_entities(document):
    if len(document.ents) == 0:
        raise Exception("No entities are found!")
    else:
        for entity in document.ents:
            print(str(entity) + " ==> " + entity.text + " ==> " + str(entity.label_) + " ==> " + str(spacy.explain(entity.label_)))


# In[ ]:


document = nlp(u"Hello, how are you ?")
print(type(document.ents))
display_entities(document)


# In[ ]:


second_document = nlp(u"May I go to New York City next July to see the Empire States Building ?")
display_entities(second_document)


# In[ ]:


my_doc = nlp(u"Can I please have 700 dollars of Apple Inc. stock ?")
display_entities(my_doc)


# In[ ]:


# ent.start and ent.end looks at the index positions word by word, while ent.start_char and ent.end_char looks at 
# the index positions character by character.
for ent in my_doc.ents:
    print(ent, ent.text, ent.start, ent.end, ent.start_char, ent.end_char, ent.label_, spacy.explain(ent.label_))


# In[ ]:


my_document = nlp(u"Tesla has produced over 200 million Tesla cars in 2023 and has an expense of 300 million dollars in this production process.")


# In[ ]:


display_entities(my_document)


# In[ ]:


from spacy.tokens import Span

org = my_document.vocab.strings[u"ORG"]
print("Hash value of organization's NER abbreviation: "+str(org)+"")


# In[ ]:


new_entity = Span(my_document, 0, 1, label=org)
print(new_entity)
print(type(new_entity)) # a span object
print()


# In[ ]:


my_document_ents_list = list(my_document.ents)
my_document_ents_list.append(new_entity) # adding new entity to the existing list of named entities of the document called my_document

display_entities(my_document)


# In[ ]:


# Adding multiple phrases as NER parts
new_doc = nlp(u"Our company created a brand new washing machine."
              u" This new washing-machine is the best in show.")
print(new_doc)
print(type(new_doc))


# In[ ]:


display_entities(new_doc)


# In[ ]:


from spacy.matcher import PhraseMatcher
phraseMatcher = PhraseMatcher(nlp.vocab)
print(phraseMatcher)
print(type(phraseMatcher))


# In[ ]:


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


# In[ ]:


from spacy.tokens import Span
prod = doc.vocab.strings[u"PRODUCT"]
print(prod) # hashcode of PRODUCT named entity


# In[ ]:


new_entities = []
for match in found_matches:
    span = Span(new_doc, match[1], match[2], label=prod) # gets the span of the match
    new_entities.append(span)
    
print(new_entities)
print(type(new_entities))

print()
print()

new_doc_ents_lst = list(new_doc.ents)
print("Before appending the new entities: "+str(new_doc_ents_lst)+"")
new_doc_ents_lst.append(new_entities)
print("After appending the new entities: "+str(new_doc_ents_lst)+"")


# In[ ]:


display_entities(new_doc)


# In[ ]:


sneaker_document = nlp(u"I have bought this sneaker for 120 dollars, however it is now discounted by 15 dollars.")
print(sneaker_document)
print(type(sneaker_document))


# In[ ]:


number_of_money_entities = len([entity for entity in sneaker_document.ents if entity.label_ == "MONEY"])

print([entity.label_ for entity in sneaker_document.ents]) # gets the string representations of the entity labels in the sneaker_document
print([entity.label for entity in sneaker_document.ents]) # gets the numeric ids of the entity labels in the sneaker_document
print([entity for entity in sneaker_document.ents])
print([entity for entity in sneaker_document.ents if entity.label_ == "MONEY"])
print(number_of_money_entities)


# In[ ]:





# In[ ]:





# In[ ]:




