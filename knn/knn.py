#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model, datasets, neighbors
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# In[2]:


def euclideDistance(row1, row2):
    return np.sum((row1-row2) ** 2)


# In[3]:


class KNearestNeighbors:
    
    def __init__(self, k = 3):
        
        self.k = k
    
    def fit(self, data_train, label_train):
        self.data_train = data_train
        self.label_train = label_train
        
    def getNeighbors(self, data_test):
        
        # Get distance list from data test to each data train
        distance = []
        for i in range(len(self.data_train)):
            d = euclideDistance(self.data_train[i], data_test)
            distance.append((self.data_train[i], self.label_train[i], d))
            
        # Sort to get smallest distance data
        distance.sort(key=lambda tup: tup[-1])
        
        # Select k smallest data
        return distance[:self.k-1][:1]
            
        
    def predict(self, data_test):
        
        label_pred = []
        
        for data in data_test:
            neighbor = self.getNeighbors(data)
        
            neighbor_label = [row[1] for row in neighbor]
        
            label_pred.append(max(set(neighbor_label), key=neighbor_label.count))
            
        return np.array(label_pred)
        
        


# In[4]:


np.random.seed(3)


# In[5]:


iris = datasets.load_iris()


# In[6]:


X = iris.data[:, :2]
y = iris.target


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)


# In[8]:


model = KNearestNeighbors(k = 10)


# In[9]:


model.fit(X_train, y_train)


# In[10]:


y_pred = model.predict(X_test)
y_pred[:15]


# In[11]:


y_test[:15]


# In[12]:


model1 = neighbors.KNeighborsClassifier(n_neighbors=10)


# In[13]:


model1.fit(X_train, y_train)


# In[14]:


y_pred1 = model1.predict(X_test)


# In[15]:


y_pred1[:15]


# In[16]:


accuracy_score(y_test, y_pred)


# In[17]:


accuracy_score(y_test, y_pred1)


# In[ ]:




