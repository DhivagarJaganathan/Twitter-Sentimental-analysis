#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("C:\\Users\\DHIVAGAR\\Desktop\\Data files\\sentiment_tweets3.csv")


# In[3]:


df.head()


# In[4]:


df.rename(columns={"label (depression result)":"label","message to examine":"tweet"},inplace=True)


# In[5]:


df['label'].value_counts()


# In[6]:


#library that contains punctuation
import string
string.punctuation


# In[7]:


#defining the function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree
#storing the puntuation free text
df['clean_msg']= df['tweet'].apply(lambda x:remove_punctuation(x))
df.head()


# In[8]:


#lower the letters 


# In[9]:


df['msg_lower']= df['clean_msg'].apply(lambda x: x.lower())


# In[10]:


df.head()


# In[11]:


#defining function for tokenization
import re
def tokenization(text):
    tokens = re.split('W+',text)
    return tokens
#applying function to the column
df['msg_tokenied']= df['msg_lower'].apply(lambda x: tokenization(x))


# In[12]:


df.head()


# In[13]:


get_ipython().system('pip install stopwords')


# In[14]:


#importing nlp library
import nltk
nltk.download('stopwords')


# In[15]:


#Stop words present in the library
#Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')
stopwords[0:10]


# In[16]:


#defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output


# In[17]:


#applying the function
df['no_stopwords']= df['msg_tokenied'].apply(lambda x:remove_stopwords(x))


# In[18]:


df.head()


# In[19]:


#importing the Stemming function from nltk library
from nltk.stem.porter import PorterStemmer
#defining the object for stemming
porter_stemmer = PorterStemmer()


# In[20]:


#defining a function for stemming
def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text
df['msg_stemmed']=df['no_stopwords'].apply(lambda x: stemming(x))


# In[21]:


df.head()


# In[22]:


from nltk.stem import WordNetLemmatizer
#defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()


# In[23]:


import nltk
nltk.download('wordnet')


# In[24]:


#defining the function for lemmatization
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text
df['msg_lemmatized']=df['no_stopwords'].apply(lambda x:lemmatizer(x))


# In[25]:


df.head()


# In[26]:


df["clean_str"]=df["msg_lemmatized"]. apply(str)


# In[27]:


from textblob import TextBlob


# In[28]:


def getsubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


# In[29]:


def getpolarity(text):
    return TextBlob(text).sentiment.polarity


# In[30]:


df["Subjectivity"]=df["clean_str"].apply(getsubjectivity)
df["Polarity"]=df["clean_str"].apply(getpolarity)


# In[31]:


df.head()


# In[32]:


def getAnalysis(score):
    if score<0:
        return "negative"
    elif score==0:
        return "netural"
    else:
        return "positive"


# In[33]:


df["Analysis"]=df["Polarity"].apply(getAnalysis)


# In[34]:


df.head()


# In[35]:


import matplotlib.pyplot as plt


# In[36]:


df["Analysis"].value_counts()
plt.title("Sentiment analysis")
plt.xlabel("Sentiment")
plt.ylabel("counts")
df["Analysis"].value_counts().plot(kind="bar")
plt.show()


# In[37]:


df=df[df["Polarity"]!=0]


# In[38]:


import numpy as np


# In[39]:


df["Positive Rated"]=np.where(df["Polarity"]<0,0,1)


# In[40]:


df["Positive Rated"].value_counts()


# In[41]:


df.head()


# In[42]:


import seaborn as sns


# In[43]:


sns.countplot(df["Positive Rated"])


# In[44]:


from sklearn.model_selection import train_test_split


# In[45]:


x_train,x_test,y_train,y_test=train_test_split(df["clean_str"],df["Positive Rated"],random_state=50)


# In[46]:


print(x_train)


# In[47]:


print(y_train)


# In[48]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[49]:


vect=TfidfVectorizer().fit(x_train)


# In[50]:


len(vect.get_feature_names())


# In[51]:


x_train_vectorized=vect.transform(x_train)


# In[52]:


from sklearn.linear_model import LogisticRegression


# In[53]:


model=LogisticRegression()


# In[54]:


model.fit(x_train_vectorized,y_train)


# In[55]:


pred=model.predict(vect.transform(x_test))


# In[56]:


print(pred)


# In[57]:


from sklearn.metrics import roc_auc_score,confusion_matrix,accuracy_score,classification_report


# In[58]:


print("AUC:",roc_auc_score(y_test,pred))


# In[59]:


print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
print(accuracy_score(y_test,pred))


# In[ ]:




