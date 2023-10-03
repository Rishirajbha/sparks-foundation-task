#!/usr/bin/env python
# coding: utf-8

# In[119]:


import pandas as pd
import numpy as np


# In[120]:


# Importing  iris dataset

iris= pd.read_csv("Iris.csv")
iris


# ## Workshop of Decission Tree

# In[121]:


iris.info()


# #### Checking Duplicate records

# In[122]:


iris[iris.duplicated()]


# In[123]:


iris.head(2)

## Id feature is not required for modeling so eleminating this
# In[124]:


iris.drop(columns=['Id'],inplace= True)
iris.head(2)


# ### Encoding 

# In[125]:


iris['Species'].unique()


# #### Using label encoding for "Species" feature

# In[126]:


from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder instance
label_encoder = LabelEncoder()

# Fit and transform the 'Species' column to encode it
iris['Species'] = label_encoder.fit_transform(iris['Species'])
iris['Species'].unique()


# In[127]:


iris.head(2)


# In[128]:


# iris['Species']= iris['Species'] = iris['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})


# In[129]:


iris['Species'].unique()


# In[130]:


iris.head(1)


# ### Spliting X & Y

# In[132]:


x= iris.drop(columns=['Species'])
y=iris['Species']


# In[133]:


x.head()


# In[134]:


y.head()


# In[135]:


x.shape


# In[136]:


y.shape


# ## Spliting For Train and Test

# In[162]:


from sklearn.model_selection import train_test_split


# In[163]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)


# In[171]:


x_test.shape


# In[ ]:





# ### Workshop - Decision Trees
# 
# This workshop deals with understanding the working of decision trees.

# In[174]:


#  decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier(max_depth = 10, max_features = 7)
dtree.fit(x_train,y_train)

print('Decision Tree Classifer Created')


# In[177]:


y_pred= dtree.predict(x_test)
y_pred


# In[179]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# ### Cross Validation

# In[183]:


# Perform cross-validation and calculate accuracy
from sklearn.model_selection import cross_val_score
scores = cross_val_score(dtree, x_train, y_train, cv=5, scoring='accuracy')
mean_accuracy = scores.mean()
print("Cross-Validation Mean Accuracy:", mean_accuracy)


# ### To Check Over Fitting in Decision Tree

# In[187]:


y_pred_train = dtree.predict(x_train)
accuracy_score(y_train, y_pred_train)

# Over fitting issue is there accuracy on train data should not be 1
# ## visualization the Decision Tree to understand it better

# In[172]:


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Create and fit the Decision Tree Classifier
dtree = DecisionTreeClassifier()
dtree.fit(x, y)

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(dtree, feature_names=x_train.columns.tolist(), filled=True, rounded=True)
plt.show()


# ## Project Complited

# In[ ]:





# In[ ]:




