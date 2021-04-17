#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


dataset=pd.read_csv('daily_csv.csv')


# In[3]:


dataset.tail()


# In[4]:


dataset.head(10)


# In[5]:


dataset['year'] = pd.DatetimeIndex(dataset['Date']).year
dataset['month'] = pd.DatetimeIndex(dataset['Date']).month
dataset['day'] = pd.DatetimeIndex(dataset['Date']).day


# In[6]:


dataset.head()


# In[7]:


dataset.corr()


# In[8]:


dataset.drop('Date', axis=1, inplace=True)


# In[9]:


dataset.isnull().any()


# In[10]:


dataset.info()


# In[11]:


dataset['Price'].fillna(dataset['Price'].mean(),inplace=True)


# In[12]:


dataset.isnull().any()


# In[13]:


#import the matplotlib library
import matplotlib.pyplot as plt
#plot size
fig=plt.figure(figsize=(5,5))
plt.scatter(dataset['day'],dataset['Price'],color='blue')
#Set the label for the x-axis.
plt.xlabel('Day')
#Set the label for the y-axis.
plt.ylabel('Price')
#Set a title for the axes.
plt.title('PRICE OF NATURAL GAS ON THE BASIS OF DAYS OF A MONTH')
#Place a legend on the axes.
plt.legend()


# In[14]:


import matplotlib.pyplot as plt
plt.bar(dataset['month'],dataset['Price'],color='green')
plt.xlabel('Month')
plt.ylabel('Price')
plt.title('PRICE OF NATURAL GAS ON THE BASIS OF MONTHS OF A YEAR')
plt.legend()


# In[15]:


import seaborn as sns
sns.lineplot(x='year',y='Price',data=dataset,color='red')


# In[16]:


fig=plt.figure(figsize=(8,4))
plt.scatter(dataset['year'],dataset['Price'],color='purple')
plt.xlabel('Month')
plt.ylabel('Price')
plt.title('PRICE OF NATURAL GAS ON THE BASIS OF MONTHS OF A YEAR')
plt.legend()


# In[17]:


import seaborn as sns
sns.pairplot(dataset)
plt.show()


# In[19]:


x=dataset.iloc[:,1:4].values #inputs
y=dataset.iloc[:,0:1].values #output price only


# In[20]:


x


# In[21]:


y


# In[22]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,
                        test_size=0.2,random_state=0)


# In[23]:


x_train.shape


# In[24]:


x_test.shape


# In[25]:


#importing linear regression from scikit learn library
from sklearn.linear_model import LinearRegression
#mlr is object of LinearRegression
mlr=LinearRegression()
#trainig the model using fit method
mlr.fit(x_train,y_train)


# In[26]:


y_pred=mlr.predict(x_test)


# In[27]:


y_pred


# In[28]:


y_test


# In[29]:


from sklearn.metrics import r2_score
accuracy=r2_score(y_test,y_pred)


# In[30]:


accuracy


# In[31]:


#import decision tree regressor
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
#fitting the model or training the model
dtr.fit(x_train,y_train)


# In[32]:


y_pred=dtr.predict(x_test)


# In[33]:


y_pred


# In[34]:


y_test


# In[35]:


from sklearn.metrics import r2_score
dtraccuracy=r2_score(y_test,y_pred)


# In[36]:


dtraccuracy


# In[37]:


dataset.head()


# In[38]:


y_p=dtr.predict([[2005,12,4]])


# In[39]:


y_p


# In[40]:


y_p=dtr.predict([[1997,1,7]])


# In[41]:


y_p


# In[42]:


import pickle
pickle.dump(dtr,open('gas.pkl','wb'))


# In[ ]:




