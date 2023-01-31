#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[20]:


rw= pd.read_csv("winequality.csv")


# In[21]:


rw.head()


# In[22]:


rw.info()


# In[5]:


rw["quality"].value_counts()


# In[6]:


plt.figure(figsize = (15,10))
sns.heatmap(rw.corr(), annot = True, cmap = 'magma')
plt.show()


# In[23]:


rw=rw.drop(['quality'],axis=1)


# In[24]:


rw.head()


# In[26]:


from sklearn.cluster import KMeans


# In[29]:


wcss = []
for each in range(1,11):
    
    #print(each)
    
    kmeans = KMeans(n_clusters=each, init='k-means++', random_state=123)
    kmeans.fit(rw)
    wcss_value  = kmeans.inertia_
    wcss.append(wcss_value)


# In[30]:


wcss


# In[33]:


x_axis= [1,2,3,4,5,6,7,8,9,10]
y_axis= wcss
plt.plot(x_axis,y_axis)


# In[36]:


kmeans = KMeans(n_clusters=3, init='k-means++', random_state=123)


# In[37]:


kmeans.fit(rw)


# In[38]:


clus=kmeans.predict(rw)
clus


# In[39]:


rw["cluster"]=clus


# In[40]:


rw


# In[47]:


import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(rw, method='ward'))


# In[ ]:




