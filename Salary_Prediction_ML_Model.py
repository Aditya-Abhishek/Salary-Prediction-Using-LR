#!/usr/bin/env python
# coding: utf-8

# # Salary Prediction (Regression Problem) Machine Learning

# ## Import Required libraries

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , mean_squared_error


# ## Load Data

# In[6]:


data = pd.read_csv('Salary.csv')
data


# # Perform EDA

# ## Chekc null value is present or not

# In[7]:


data.isnull().sum()


# In[8]:


data.info()


# In[9]:


data.describe()


# ## Visualize data

# In[10]:


plt.scatter( data['YearsExperience'] ,data['Salary'] )
plt.xlabel('Year of Exp')
plt.ylabel('Salary')
plt.show()


# ## Prepare data

# In[11]:


X = data.drop('Salary',axis=1)
y = data['Salary']


# In[12]:


X.shape , y.shape


# In[13]:


X


# In[27]:


y


# ## Split data into train and test

# In[14]:


X_train , X_test , Y_train , Y_test = train_test_split(X,y,random_state=101,test_size=0.2)
X_train.shape , X_test.shape , Y_train.shape , Y_test.shape


# ## Define LinearRegression Model

# In[15]:


lr = LinearRegression()
lr.fit(X_train, Y_train)


# ## Test model

# In[16]:


pred = lr.predict(X_test)
pred


# In[17]:


Y_test


# ## Check Actual data , Predicted data and difference between the Actual and Predicted data

# In[18]:


diff = Y_test - pred


# In[19]:


pd.DataFrame(np.c_[Y_test , pred , diff] , columns=['Actual','Predicted','Difference'])


# ## Visualize Model, that how it is performing on training data

# In[20]:


plt.scatter(X_train , Y_train , color='blue')
plt.plot(X_train ,lr.predict(X_train),color='red')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel("Salary")
plt.show()


# ## Visualize Model, that how it is performing on testing data

# In[21]:


plt.scatter(X_test , Y_test,color='blue')
plt.plot(X_test ,lr.predict(X_test) ,color='red')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel("Salary")
plt.show()


# ## Evaluate

# In[22]:


lr.score(X_test , Y_test)


# In[23]:


rmse = np.sqrt(mean_squared_error(Y_test,pred))
r2 = r2_score(Y_test,pred)


# In[24]:


rmse , r2


# # Test on the custom data

# In[25]:


exp = 3
lr.predict([[exp]])[0]
print(f"Salary of {exp} year experience employee = {int(lr.predict([[exp]])[0])} thousands")


# In[26]:


exp = 5
lr.predict([[exp]])[0]
print(f"Salary of {exp} year experience employee = {int(lr.predict([[exp]])[0])} thousands")


# # Thank You !!!!!!!!!!
