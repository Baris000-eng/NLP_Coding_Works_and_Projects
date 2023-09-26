#!/usr/bin/env python
# coding: utf-8

# In[66]:


import spacy

nlp = spacy.load("en_core_web_sm")
print(nlp)


# In[67]:


from spacy.matcher import Matcher 
print(nlp.vocab)
matcher = Matcher(nlp.vocab)


# In[68]:


print(matcher, type(matcher))
print(matcher)
print(type(matcher))


# In[69]:


# Patterns to be found:
# SolarPower
# Solar-power
# Solar power

## Token patterns to perform rule-based matching 

#SolarPower
# The first pattern checks when we transform the token to its lowercase version, if it will be same as 'solarpower'.
first_pattern = [{'LOWER': 'solarpower'}]

#Solar-power
second_pattern = [{'LOWER': 'solar'}, {'IS_PUNCT':True}, {'LOWER': 'power'}]

#Solar power
third_pattern = [{'LOWER': 'solar'}, {'LOWER': 'power'}]


# In[70]:


matcher.add("SolarPower", [first_pattern])
matcher.add("Solar-power", [second_pattern])
matcher.add("Solar power", [third_pattern])


# In[71]:


document = nlp(u'The Solar Power industry continues to grow as solarpower increases. Solar-power is a crucial resource.')
found_matches = matcher(document)
print(found_matches)

# In the output of this cell, each tuple will contain the match id, starting index of the match and the ending 
# index of the match in sequence. The starting and ending indexes are at the token level. In other words; the 
# index 0 belongs to the first word, the index 1 belongs to the second word, and so forth.


# In[72]:


for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id] # get string representation
    matched_span = document[start:end] # get the matched span
    print(match_id, string_id, start, end, matched_span.text)


# In[73]:


matcher.remove('SolarPower') # Remove 'SolarPower' from the matcher object


# In[74]:


overlapping_part = matcher(document)
print(overlapping_part)


# In[75]:


# 'OP': '*' allows the pattern to match zero or more times. For the pattern2, it allows us to put 
# any amount of punctuation in the middle.l
pattern1 = [{'LOWER': 'solarpower'}]
pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP': '*'}, {'LOWER': 'power'}]


# In[76]:


my_matcher = Matcher(nlp.vocab)
my_matcher.add('solarpower', [pattern1]) # adding pattern1 to the matcher object
my_matcher.add('Solar-Power', [pattern2]) # adding pattern2 to the matcher object


# In[77]:


new_document = nlp(u'Solar--power can be solarpower, Solar Power, or solar-power.')
matches = matcher(new_document)
print(matches)


# In[121]:


# Phrase Matching 
from spacy.matcher import PhraseMatcher 

phraseMatcher = PhraseMatcher(nlp.vocab) # creating a phrase matcher instance

with open('reaganomics.txt') as file:
    content = file.read()
    third_doc = nlp(content)
    
print(third_doc)


# In[122]:


phrases = ["supply-side economics", "voodo economics", "free-market economics", "trickle-down economics"]


# In[123]:


phrase_patterns = []
for phrase in phrases:
    doc = nlp(phrase)
    phrase_patterns.append(doc)


# In[124]:


phrase_patterns


# In[125]:


print(phrase_patterns)


# In[126]:


print(type(phrase_patterns))
print(type(phrase_patterns[0])) # An element of the list of phrase patterns is of type 'spacy Doc'.


# In[134]:


# Instead of a single pattern as in the case of Matcher, it can take a list of patterns as an argument.
phraseMatcher.add('EconMatcher', phrase_patterns) 
matches_lst = phraseMatcher(third_doc)
print(matches_lst)


# In[137]:


for match_id, initial_index, end_index in matches_lst:
    string_id = nlp.vocab.strings[match_id] # get the string representation of the match id
    matched_span = third_doc[initial_index:end_index] # get the matched span
    print(match_id, string_id, initial_index, end_index, matched_span, matched_span.text)


# In[ ]:





# In[ ]:





# In[ ]:




