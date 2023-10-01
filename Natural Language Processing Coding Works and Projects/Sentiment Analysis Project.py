#!/usr/bin/env python
# coding: utf-8

# In[112]:


# Import spacy and load large english language library 
import spacy

nlp = spacy.load('en_core_web_md')
print(nlp)


# In[113]:


# choose the words to be compared and obtain the vectors


word1_vec = nlp.vocab['wolf'].vector
word2_vec = nlp.vocab['dog'].vector
word3_vec = nlp.vocab['cat'].vector


# In[114]:


# cosine similarity function
import numpy as np
from scipy.spatial.distance import cosine

import numpy as np


# The cosine similarity between two vectors is the dot product of the vectors divided by the multiplication of the 
# lengths of them.
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    similarity = dot_product / (norm_vec1 * norm_vec2)
    
    return similarity


# In[115]:


# write the expression for the vector arithmetic
# For example: new_vector = word1 - word2 + word3
new_word_vector = word1_vec - word2_vec + word3_vec
print(new_word_vector)


# In[116]:


# list the top 20 closest vectors in the vocabulary to the result of above vector arithmetic

calculated_similarities = []
for word in nlp.vocab:
    if word.has_vector:
        if word.is_lower:
            if word.is_alpha:
                similarity = cosine_similarity(new_word_vector, word.vector)
                calculated_similarities.append((word, similarity))
        
print(calculated_similarities)
        


# In[117]:


# sort the calculated similarities in a descending order and based on the similarity socres
sorted_similarities = sorted(calculated_similarities, key = lambda sim: sim[1], reverse=True)
print(sorted_similarities)


# In[118]:


# top 20 similar words in the vocabulary
top20_similar_words = [word[0].text for word in sorted_similarities[:20]]
print(top20_similar_words)


# In[119]:


def vector_math(s1, s2, s3):
    word1_vec = nlp.vocab[s1].vector
    word2_vec = nlp.vocab[s2].vector
    word3_vec = nlp.vocab[s3].vector
    
    new_vec = word1_vec - word2_vec + word3_vec
    
    computed_sims = []
    for word in nlp.vocab: 
        if word.has_vector:
            if word.is_lower:
                if word.is_alpha:
                    sim = cosine_similarity(new_vec, word.vector)
                    computed_sims.append((word, sim))
                    
                    
                    
    sorted_sims = sorted(computed_sims, key = lambda sim: sim[1], reverse=True)
    return [word[0].text for word in sorted_sims[:20]]



vector_math('king', 'man', 'women')
    
    


# In[120]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer 

sia = SentimentIntensityAnalyzer()

review = "This movie is absolutely awful. This is the WORST movie that I have ever seen."
review_polarity = sia.polarity_scores(review)
print(review_polarity)


# In[121]:


def review_rating(review: str): 
    scores = sia.polarity_scores(review)
    
    if scores['compound'] == 0:
        return "Neutral"
    elif scores['compound'] > 0:
        return "Positive"
    return "Negative"


# In[124]:


print("The review rating result is: "+str(review_rating(review))+"")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




