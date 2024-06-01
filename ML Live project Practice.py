#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_excel(r"C:\Users\DIVYA SHAH\Downloads\DS DOWNLOADS\MACHINE LEARNING\ML Live Flight Fare Resourses16963295320-1.xlsx")
data


# In[3]:


data.isnull().any()


# In[5]:


data.isnull().sum()


# In[6]:


import numpy as np


# In[7]:


data[data["Route"].isnull()==True]


# In[8]:


data.drop(9039,inplace=True)


# In[9]:


data.isnull().any()


# In[10]:


data.info()


# In[11]:


data.drop(["Route","Additional_Info"],axis=1,inplace=True)


# In[12]:


data.head(2)


# In[17]:


data["day"] = pd.to_datetime(data["Date_of_Journey"]).dt.day
data["month"] = pd.to_datetime(data["Date_of_Journey"]).dt.month


# In[18]:


data.drop("year",axis=1,inplace=True)


# In[19]:


data.drop("Date_of_Journey",axis=1,inplace=True)


# In[20]:


data.head(3)


# In[21]:


import warnings
warnings.filterwarnings("ignore")


# In[23]:


data["dep_hour"] = pd.to_datetime(data["Dep_Time"]).dt.hour
data["dep_minute"] = pd.to_datetime(data["Dep_Time"]).dt.minute


# In[24]:


data.head(3)


# In[27]:


data["Arr_hour"] = pd.to_datetime(data["Arrival_Time"]).dt.hour
data["Arr_minute"] = pd.to_datetime(data["Arrival_Time"]).dt.minute


# In[28]:


data.drop(["Dep_Time","Arrival_Time"],axis=1,inplace=True)


# In[29]:


data.head(3)


# In[40]:


lis=list(data["Duration"])
lis1=[]


# In[41]:


for i in lis:
    
    if (len(i.split(" "))==1):
    
        if "h" in i: 
            i=i+" 0m"    
        else:
            i="0h "+i
           
    lis1.append(i)


# In[42]:


data["Duration"]=lis1


# In[43]:


data.head()


# In[44]:


data["Total_Stops"].unique()


# In[45]:


data["Total_Stops"].replace(['non-stop', '1 stop','2 stops', '3 stops', '4 stops'],[0,1,2,3,4],inplace=True)


# In[46]:


data


# In[48]:


data["Airline"].nunique()


# In[49]:


data["Source"].nunique()


# In[50]:


data["Destination"].nunique()


# In[51]:


from sklearn.preprocessing import LabelEncoder


# In[53]:


enc = LabelEncoder()


# In[54]:


data[["Airline","Source","Destination"]] = data[["Airline","Source","Destination"]].apply(enc.fit_transform)


# In[55]:


data.head()


# In[56]:


data.info()


# In[57]:


data


# In[60]:


data["dur_hour"]=data["Duration"].str.split(" ").str[0].replace("[h]","",regex=True)
data["dur_min"] = data["Duration"].str.split(" ").str[1].replace("[m]","",regex=True)


# In[61]:


data


# In[62]:


data["Duration"].str.split(" ")[0]


# In[63]:


data.drop("Duration",axis=1,inplace=True)


# In[64]:


data.info()


# In[65]:


data[["dur_hour","dur_min"]] = data[["dur_hour","dur_min"]].astype(int)


# In[66]:


x = data.drop("Price",axis=1)


# In[67]:


y = data["Price"]


# In[68]:


from sklearn.model_selection import train_test_split


# In[69]:


x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=0)


# In[70]:


from sklearn.linear_model import LinearRegression


# In[71]:


model = LinearRegression()


# In[72]:


model.fit(x_train,y_train)


# In[74]:


# In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
# On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.


# In[75]:


model.score(x_train,y_train)


# In[76]:


model.score(x_test,y_test)


# In[80]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


# In[82]:


lis =[DecisionTreeRegressor,RandomForestRegressor,SVR]


# In[84]:


for i in lis:
    model=i()
    model.fit(x_train,y_train)
    print(i)
    print(model.score(x_train,y_train))
    print(model.score(x_test,y_test))


# In[87]:


from sklearn.model_selection import KFold, cross_val_score


# In[88]:


name=[]
score=[]
for i in lis:
    kf=KFold(5)
    cvr=cross_val_score(i(),x,y,cv=kf)
    name.append(i)
    score.append(cvr)


# In[89]:


name


# In[90]:


score


# In[91]:


for i in range(len(name)):
    print(name[i],"/t",score[i].mean())


# In[93]:


model = RandomForestRegressor()
model.fit(x_train,y_train)
print(model.score(x_train,y_train))
print(model.score(x_test,y_test))


# In[94]:


import pickle


# In[95]:


import joblib


# In[97]:


joblib.dump(model,r"C:\Users\DIVYA SHAH\Downloads\DS DOWNLOADS\MACHINE LEARNING\output.pkl")
model1=joblib.load(r"C:\Users\DIVYA SHAH\Downloads\DS DOWNLOADS\MACHINE LEARNING\output.pkl")


# In[98]:


model1.score(x,y)


# In[ ]:




