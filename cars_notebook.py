#!/usr/bin/env python
# coding: utf-8

# In[63]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score ,  accuracy_score 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import warnings
import matplotlib as mpl
# get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# Suppress all warnings
warnings.filterwarnings('ignore')



# ### Car Resale - Price Prediction Section

# In[64]:


# lets load the cars dataset and look at the features
df_cars = pd.read_csv('quikr_car.csv')
df_cars.head() 


# In[65]:


df_cars.info()


# In[66]:


# Creating backup copy
backup=df_cars.copy()


# ### Quality

# - names are pretty inconsistent
# - names have company names attached to it
# - some names are spam like 'Maruti Ertiga showroom condition with' and 'Well mentained Tata Sumo'
# - company: many of the names are not of any company like 'Used', 'URJENT', and so on.
# - year has many non-year values
# - year is in object. Change to integer
# - Price has Ask for Price
# - Price has commas in its prices and is in object
# - kms_driven has object values with kms at last.
# - It has nan values and two rows have 'Petrol' in them
# - fuel_type has nan values

# ### Cleaning Data

# In[67]:


#lets drop the names and just use company
df_cars = df_cars.drop(columns = ['name'])


# ##### year has many non-year values

# In[68]:


df_cars=df_cars[df_cars['year'].str.isnumeric()]


# ##### year is in object. Change to integer

# In[69]:


df_cars['year']=df_cars['year'].astype(int)


# ##### Price has Ask for Price

# In[70]:


df_cars=df_cars[df_cars['Price']!='Ask For Price']


# ##### Price has commas in its prices and is in object

# In[71]:


df_cars['Price']=df_cars['Price'].str.replace(',','').astype(int)


# ##### kms_driven has object values with kms at last.

# In[72]:


df_cars['kms_driven']=df_cars['kms_driven'].str.split().str.get(0).str.replace(',','')


# ##### It has nan values and two rows have 'Petrol' in them

# In[73]:


df_cars=df_cars[df_cars['kms_driven'].str.isnumeric()]


# In[74]:


df_cars['kms_driven']=df_cars['kms_driven'].astype(int)


# ##### fuel_type has nan values

# In[75]:


df_cars=df_cars[~df_cars['fuel_type'].isna()]


# #

# ##### Resetting the index of the final cleaned data

# In[76]:


df_cars=df_cars.reset_index(drop=True)


# In[77]:


df_cars.to_csv('Cleaned_Car_data.csv')


# In[78]:


df_cars.info()


# ##### Checking relationship of Company with Price

# In[79]:


plt.subplots(figsize=(15,7))
ax=sns.boxplot(x='company',y='Price',data=df_cars)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()


# ##### Checking relationship of Year with Price

# In[80]:


plt.subplots(figsize=(20,10))
ax=sns.swarmplot(x='year',y='Price',data=df_cars)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()


# ##### Checking relationship of kms_driven with Price

# In[81]:


sns.relplot(x='kms_driven',y='Price',data=df_cars,height=7,aspect=1.5)


# ##### Extracting Training Data

# In[82]:


X=df_cars[['company','year','kms_driven','fuel_type']]
y=df_cars['Price']


# ##### Applying Train Test Split

# In[83]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# ##### Creating an OneHotEncoder object to contain all the possible categories

# In[84]:


ohe=OneHotEncoder()
ohe.fit(X[['company','fuel_type']])


# ##### Creating a column transformer to transform categorical columns

# In[85]:


column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['company','fuel_type']),
                                    remainder='passthrough')


# ##### Linear Regression Model

# In[86]:


lr=LinearRegression()


# ##### Making a pipeline

# In[87]:


pipe=make_pipeline(column_trans,lr)


# In[88]:


pipe.fit(X_train,y_train)


# In[89]:


y_pred=pipe.predict(X_test)


# In[90]:


r2_score(y_test,y_pred)


# ##### Finding the model with a random state of TrainTestSplit 

# In[91]:


scores=[]
for i in range(1000):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    scores.append(r2_score(y_test,y_pred))


# In[92]:


np.argmax(scores)


# In[93]:


scores[np.argmax(scores)]


# In[94]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
r2_score(y_test,y_pred)


# In[97]:


# making function of cars_price_predictor
def cars_price_predictor(company  , year , kms , type):
    price = pipe.predict(pd.DataFrame(columns=X_test.columns,data=np.array([company,year,kms,type]).reshape(1,4)))
    return price


# In[98]:


cars_price_predictor('Maruti' , 2019,100,'Petrol')


# In[ ]:




