#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_iris


# In[45]:


data=load_iris()


# In[ ]:





# In[46]:


X=data.data
Y=data.target
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,shuffle=True,random_state=700)


# In[47]:


dt =tree.DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
predictions=dt.predict( X_test)
print('Accuracy= ',accuracy_score(y_test,predictions)*100)


# In[48]:


from sklearn import tree
tree.plot_tree(dt)


# In[50]:


tree.plot_tree(dt,
               feature_names = data.feature_names, 
               class_names=data.target_names,
               rounded=True, 
               filled = True);


# In[ ]:




