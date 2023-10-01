#!/usr/bin/env python
# coding: utf-8

# In[1]:


# VADER (Valence Aware Dictionary for Sentiment Reasoning) is a model used for text sentiment analysis that is 
# sensitive to both polarities (positive/negative), the neutral texts and intensity (strength) of the emotion.

# - Vader Model is available in the NLTK package and it can be applied to unlabeled text data. 


# Vader Sentiment Analysis primarily relies on a dictionary which maps lexical features to emotion intensities
# called sentiment scores.

# The sentiment score of a text can be obtained by summing up the intensity of each word in the text.


# Document Sentiment Score: We grab all words in a document. Then, we convert each word to a positive or 
# negative value. Finally, we sum up all values we find. At the end, we will find the document sentiment 
# score.

# For instance, the words such as 'love', 'like', 'enjoy', and 'happy' all convey a positive sentiment.

# VADER is intelligent enough to understand basic contexts of these words, such as 'did not love' as a 
# negative sentiment. It also understands punctuation and capitalization, like "LOVE!!!"

# VADER is also smart enough to understand that the word 'LOVE!!!!!' conveys more positive information than 
# the word 'love'.

# Sentiment Analysis on raw text is always challenging due to a variety of possible factors: 

# 1-) Positive and Negative Sentiment in the same text data
# 2-) Using positive words in a negative way (iğnelemek)


# In[2]:


# necessary imports 
import nltk


# In[3]:


nltk.download('vader_lexicon')


# In[4]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer


# 'Vader Sentiment Intensity Analyzer' takes a string and returns a dictionary of scores in four categories:
# 1-) Negative score
# 2-) Neutral score 
# 3-) Positive score 
# 4-) Compound score 

# Compound score is computed by normalizing the negative, neutral, and positive scores.

sentiment_intensity_analyzer = SentimentIntensityAnalyzer()
print(sentiment_intensity_analyzer)
print(type(sentiment_intensity_analyzer))


# In[5]:


string = "This is a good movie."
polarity_scores = sentiment_intensity_analyzer.polarity_scores(string)
print(polarity_scores)


# In[6]:


neg_score = polarity_scores['neg']
pos_score = polarity_scores['pos']
neu_score = polarity_scores['neu']
comp_score = polarity_scores['compound']

print("The sentiment scores for the text named 'string' is as below: ")
print()
print("The negative score: "+str(neg_score)+"")
print("The positive score: "+str(pos_score)+"")
print("The neutral score:  "+str(neu_score)+"")
print("The compound score: "+str(comp_score)+"")


# In[7]:


my_string = "This is the WORST movie that has ever disgraced the screen."
polarity_scores2 = sentiment_intensity_analyzer.polarity_scores(my_string)
print(polarity_scores2)


# In[8]:


negative_score = polarity_scores['neg']
positive_score = polarity_scores['pos']
neutral_score = polarity_scores['neu']
compound_score = polarity_scores['compound']

print("The sentiment scores for the text named 'my_string' is as below: ")
print()
print("The negative score: "+str(neg_score)+"")
print("The positive score: "+str(pos_score)+"")
print("The neutral score: "+str(neu_score)+"")
print("The compound score: "+str(comp_score)+"")

# When the compound score is 0, then it means that the text is neutral. When the compound score is smaller than 
# zero, then it means that the text is negative. When the compound score is greater than zero, then it means that 
# the text is positive. 


# In[9]:


import pandas as pd

# reading the tsv file called 'amazonreviews.tsv'
amazon_reviews_df = pd.read_csv('amazonreviews.tsv', sep='\t')

print("The Amazon reviews data frame is as below:")
print()
print(amazon_reviews_df)


# In[10]:


# Displaying the head, in other words the first 5 records, of the Amazon reviews data frame
amazon_reviews_df.head()


# In[11]:


print("The first 5 records (head) of the Amazon reviews data frame: ")
print()
print(amazon_reviews_df.head())


# In[12]:


# Displaying the tail, in other words the last 5 records, of the Amazon reviews data frame
amazon_reviews_df.tail()


# In[13]:


print("The last 5 records (tail) of the Amazon reviews data frame: ")
print()
print(amazon_reviews_df.tail())


# In[14]:


row_num, col_num = amazon_reviews_df.shape
print("There are "+str(row_num)+" rows and "+str(col_num)+" columns in the Amazon reviews data frame.")
print()
print("The total number of rows: "+str(row_num)+"")
print("The total number of columns: "+str(col_num)+"")


# In[15]:


print("The labels in the Amazon reviews data frame: ")
print()
print(amazon_reviews_df['label'])
print()
print()
print(amazon_reviews_df['label'].value_counts()) # this gets the number of 'neg' and 'pos' labels, meaning the 
# number of negative movies and the number of positive movies.


# In[16]:


label_count = amazon_reviews_df['label'].value_counts()
negative_count = label_count['neg']
positive_count = label_count['pos']

negative_review_rate = negative_count / (negative_count + positive_count)
positive_review_rate = positive_count / (negative_count + positive_count)

print("The negative review rate in Amazon's review data frame: "+str(negative_review_rate)+"")
print("The positive review rate in Amazon's review data frame: "+str(positive_review_rate)+"")


# In[17]:


print("The reviews in the Amazon reviews data frame: ")
print()
print(amazon_reviews_df['review'])


# In[23]:


# checks the null values
amazon_reviews_df.isnull().sum() # to check whether a specific column of the data frame contains missing values.


# In[26]:


# If there were any nan values, this will be used to drop them from the 
# amazon reviews data frame.
amazon_reviews_df.dropna(inplace=True) 


# In[27]:


blanks = list()
for index, label, review in amazon_reviews_df.itertuples():
    if type(review) == str:
        if review.isspace():
            blanks.append(index)

print(blanks)


# In[34]:


amazon_reviews_df.iloc[0]['review'] # this grabs the text of the first review


# In[35]:


print(amazon_reviews_df.iloc[0]['review'])


# In[33]:


print(amazon_reviews_df.iloc[0])
print()
print()
print(amazon_reviews_df.iloc[0]['review'])
print()
print()
print(sentiment_intensity_analyzer.polarity_scores(amazon_reviews_df.iloc[0]['review']))


# In[36]:


# This adds a 'scores' column, containing the sentiment scores of each review, to the amazon reviews data frame.
amazon_reviews_df['scores'] = amazon_reviews_df['review'].apply(lambda review: sentiment_intensity_analyzer.polarity_scores(review))


# In[37]:


print(amazon_reviews_df.head())


# In[38]:


print(amazon_reviews_df.tail())


# In[39]:


amazon_reviews_df['compound'] = amazon_reviews_df['scores'].apply(lambda d: d['compound'])


# In[40]:


print(amazon_reviews_df.head())


# In[42]:


amazon_reviews_df['negative'] = amazon_reviews_df['scores'].apply(lambda d: d['neg'])


# In[43]:


amazon_reviews_df['neutral'] = amazon_reviews_df['scores'].apply(lambda d: d['neu'])
amazon_reviews_df['positive'] = amazon_reviews_df['scores'].apply(lambda d: d['pos'])


# In[44]:


print(amazon_reviews_df.head())


# In[49]:


# This applies the below prediction on the reviews:
# It says that a review is negative if the compound score of this review is smaller than 0. 
# If the compound score of the review is greater than or equal to 0, the review is positive.
amazon_reviews_df['sentiment_result'] = amazon_reviews_df['compound'].apply(lambda score: "neg" if score < 0 else "pos")


# In[50]:


print(amazon_reviews_df.head())


# In[51]:


amazon_reviews_df.drop('sent_result', axis=1, inplace=True)


# In[52]:


print(amazon_reviews_df.head())


# In[58]:


# Performance evaluation for the custom prediction made on reviews

# necessary imports
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[59]:


# accuracy score calculation
accuracy = accuracy_score(amazon_reviews_df['label'], amazon_reviews_df['sentiment_result'])
print(accuracy)


# In[60]:


# classification report 
classification_report = classification_report(amazon_reviews_df['label'], amazon_reviews_df['sentiment_result'])
print(classification_report)


# In[61]:


# confusion matrix
confusion_matrix = confusion_matrix(amazon_reviews_df['label'], amazon_reviews_df['sentiment_result'])
print(confusion_matrix)


# In[ ]:





# In[ ]:




