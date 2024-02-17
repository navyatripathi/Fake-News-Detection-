#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
get_ipython().system('')


# In[2]:


data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')


# In[3]:


data_fake.head()


# In[4]:


data_true.head()


# In[5]:


data_fake["class"] = 0
data_true['class'] = 1


# In[6]:


data_fake.shape, data_true.shape


# In[7]:


data_fake_manual_testing = data_fake.tail(10)
for i in range(23480,23470,-1):
     data_fake.drop([i], axis = 0, inplace = True)
        
data_true_manual_testing = data_true.tail(10)
for i in range(21416, 21406,-1):
    data_true.drop([i], axis = 0, inplace = True)


# In[8]:


data_fake.shape, data_true.shape


# In[9]:


data_fake_manual_testing['class'] = 0
data_true_manual_testing['class'] = 1


# In[10]:


data_fake_manual_testing.head(10)


# In[11]:


data_true_manual_testing.head(10)


# In[12]:


data_merge = pd.concat([data_fake, data_true], axis = 0)
data_merge.head(10)


# In[13]:


data_merge.columns


# In[14]:


data = data_merge.drop(['title','subject', 'date'], axis = 1)


# In[15]:


data.isnull().sum()


# In[16]:


data = data.sample(frac = 1)


# In[17]:


data.head()


# In[18]:


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '',text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# In[19]:


data['text'] = data['text'].apply(wordopt)


# In[20]:


x = data['text']
y = data['class']


# In[21]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25)


# In[22]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
x_train = vectorization.fit_transform(x_train)
x_test = vectorization.transform(x_test)


# In[23]:


from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(x_train, y_train)


# In[24]:


pred_lr = LR.predict(x_test)


# In[25]:


LR.score(x_test, y_test)


# In[26]:


print(classification_report(y_test, pred_lr))


# In[27]:


from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(x_train, y_train)


# In[28]:


pred_dt = DT.predict(x_test)


# In[29]:


DT.score(x_test, y_test)


# In[30]:


print(classification_report(y_test, pred_lr))


# In[31]:


from sklearn.ensemble import GradientBoostingClassifier

GB = GradientBoostingClassifier(random_state = 0)
GB.fit(x_train, y_train)


# In[32]:


pred_gb = GB.predict(x_test)


# In[33]:


GB.score(x_test, y_test)


# In[34]:


print(classification_report(y_test, pred_gb))


# In[35]:


from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(random_state = 0)
RF.fit(x_train, y_train)


# In[36]:


pred_rf = RF.predict(x_test)


# In[37]:


RF.score(x_test, y_test)


# In[38]:


print(classification_report( y_test, pred_rf))


# In[39]:


def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_x_test)
    pred_DT = DT.predict(new_x_test)
    pred_GB = GB.predict(new_x_test)
    pred_RF = RF.predict(new_x_test)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]), 
                                                                                                              output_lable(pred_DT[0]), 
                                                                                                              output_lable(pred_GBC[0]), 
                                                                                                              output_lable(pred_RFC[0])))


# In[40]:


news = input("Enter the news: ")
manual_testing(news)


# In[ ]:





# In[ ]:





# In[ ]:




