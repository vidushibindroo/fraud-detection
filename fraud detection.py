#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import sklearn
import numpy as np


# In[37]:


data =pd.read_csv('creditcard.csv')


# In[40]:


print(data.columns)


# In[41]:


print(data.shape)


# In[42]:


print(data.describe())


# In[43]:


data=data.sample(frac= 0.1,random_state=1)

print(data.shape)


# In[44]:


#plot histofram for the dataset
data.hist(figsize=(20,20))
plt.show()


# In[45]:


fraud=data[data['Class']==1]
valid=data[data['Class']==0]
outlier_fraction=len(fraud)/float(len(valid))
print(outlier_fraction)
print('fraud: {}'.format(len(fraud)))
print('valid: {}'.format(len(valid)))


# In[50]:


#correlation matrix
corrmat=data.corr()
fig=plt.figure(figsize=(20,20))
sns.heatmap(corrmat,vmax= .8, square= True)
plt.show()


# In[51]:


columns=data.columns.tolist()
columns=[c for c in columns if c not in ["Class"]]
target= "Class"
X=data[columns]
Y=data[target]


# In[52]:


print(X.shape)
print(Y.shape)


# In[57]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
state= 1
classifiers={"Isolation Forest": IsolationForest(max_samples =len(X), contamination=outlier_fraction, random_state= state),
             "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=outlier_fraction)}


# In[71]:


#fit the model
n_outliers= len(fraud)
for i,(clf_name,clf) in enumerate(classifiers.items()):
    if clf_name=="Local Outlier Factor":
        y_pred= clf.fit_predict(X)
        scores_pred= clf.negative_outlier_factor_
        
    else:
        clf.fit(X)
        scores_pred =clf.decision_function(X)
        y_pred= clf.predict(X)
#reshape the prediction values

y_pred[y_pred==1]=0
y_pred[y_pred==-1]=1

n_errors= (y_pred != Y).sum()


#run classificatin metric
print('{} : {}'.format(clf_name,n_errors))
print(accuracy_score(Y,y_pred))
print(classification_report(Y,y_pred))


            


# In[ ]:




