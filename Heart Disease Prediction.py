#!/usr/bin/env python
# coding: utf-8

# In[53]:


import sys
get_ipython().system('conda install --yes --prefix {sys.prefix} plotly')


# In[422]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns


import plotly
import plotly.express as px
import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot
from sklearn.metrics import accuracy_score

# import cufflinks as cf


# In[423]:


# pyo.init_notebook_mode(connected=True)
# cf.go_offline()


# In[424]:


df=pd.read_csv(r'heart.csv')


# In[425]:


df


# # Data Analysis

# In[426]:


df.columns


# In[427]:


df.dtypes


# In[428]:


info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]



for i in range(len(info)):
    print(df.columns[i]+":\t\t\t"+info[i])


# In[ ]:





# In[429]:


df['target']


# In[430]:


df.groupby('target').size()


# In[431]:


df.shape


# In[432]:


df.size


# In[433]:


df.describe()


# In[ ]:





# In[434]:


df.info()


# In[ ]:





# In[435]:


#visualization


# In[436]:


df.hist(figsize=(14,14))
plt.show()


# In[ ]:





# In[437]:


sns.barplot(df['sex'],df['target'])
plt.show()


# In[ ]:





# In[438]:


sns.barplot(df['sex'],df['age'],hue=df['target'])
plt.show()


# In[ ]:





# In[439]:


px.bar(df,df['sex'],df['target'])


# In[ ]:





# In[440]:


numerical_columns=['trestbps','chol','age','oldpeak','thalach']


# In[441]:


sns.heatmap(df[numerical_columns].corr(),annot=True,cmap='terrain',linewidths=0.1) #cmap is a colour scheme
fig=plt.gcf() #.gcf() is to convert the size to inches
fig.set_size_inches(8,6)
plt.show


# In[ ]:





# In[442]:


#create four displots
plt.figure(figsize=(12,10))
plt.subplot(221) #first column first row
sns.distplot(df[df['target']==0].age)
plt.title('Age of patients without heart disease')

plt.subplot(222)
sns.distplot(df[df['target']==1].age)
plt.title('Age of patients with heart disease')

plt.subplot(223)
sns.distplot(df[df['target']==0].thalach)
plt.title('Max Heart rate of patients without heart disease')

plt.subplot(224)
sns.distplot(df[df['target']==1].thalach)
plt.title('Max Heart rate of patients with heart disease')


# In[ ]:





# # Data Preprocessing

# In[444]:


X,y=df.loc[:,:'thal'],df['target']


# In[ ]:





# In[445]:


X


# In[446]:


y


# In[447]:


X.shape


# In[448]:


y.shape


# In[552]:


from sklearn.preprocessing import StandardScaler

std=StandardScaler().fit(X)
X_std=std.transform(X)
X


# In[555]:


X_std


# In[ ]:





# In[449]:


X.size


# In[450]:


from sklearn.model_selection import train_test_split


# In[556]:


X_train_std,X_test_std,y_train,y_test=train_test_split(X_std,y,random_state=10,test_size=0.3,shuffle=True)


# In[558]:


X_train_std.shape


# In[559]:


X_test_std.shape


# In[560]:


y_train.size


# In[561]:


y_test.size


# In[ ]:





# In[563]:


X_train_std


# # Model

# In[457]:


#decision tree classifier


# In[458]:


from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)


# In[459]:


X_test


# In[460]:


y_test


# In[461]:


prediction=dt.predict(X_test)
prediction


# In[462]:


accuracy_dt=accuracy_score(y_test,prediction)*100
accuracy_dt


# In[463]:


dt.feature_importances_


# In[465]:


def plot_features_importance(model):
    plt.figure(figsize=(8,6))
    n_features=13
    plt.barh(range(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),X)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.ylim(-1,n_features)
    
plot_features_importance(dt)


# In[ ]:





# In[466]:


df


# In[467]:


X


# In[468]:


Category=['No, You don\'t have heart disease','Yes, you have heart disease']


# In[472]:


custom_data=np.array([[63,1,3,145,233,1,0,150,0,2.7,1,2,3]])


# In[473]:


custom_data_prediction_dt=dt.predict(custom_data)


# In[474]:


custom_data_prediction_dt


# In[475]:


print(Category[int(custom_data_prediction_dt)])


# In[ ]:





# In[ ]:





# In[564]:


#KNN - K nearest Neighbour


# In[613]:


from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train_std,y_train)


# In[614]:


prediction_knn=knn.predict(X_test_std)
prediction_knn


# In[615]:


accuracy_knn=accuracy_score(y_test,prediction_knn)*100
accuracy_knn


# In[ ]:





# In[590]:


custom_data_knn=np.array([[63,1,3,145,233,1,0,150,0,2.7,1,2,3]])


# In[591]:


custom_data_knn_std=std.transform(custom_data_knn)


# In[592]:


custom_data_prediction_knn_std=knn.predict(custom_data_knn_std)


# In[593]:


int(custom_data_prediction_knn_std)


# In[594]:


print(Category[int(custom_data_prediction_knn_std)])


# In[ ]:





# In[622]:


k_range=range(1,26)
scores={}
scores_list=[]

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_std,y_train)
    prediction_knn=knn.predict(X_test_std)
    scores[k]=accuracy_score(y_test,prediction_knn)
    scores_list.append(accuracy_score(y_test,prediction_knn))


# In[623]:


scores_list


# In[625]:


plt.plot(k_range,scores_list)


# In[ ]:





# In[626]:


px.line(x=k_range,y=scores_list)


# In[ ]:





# In[627]:


algortihms=['Decision Tree','KNN']
scores=[accuracy_dt,accuracy_knn]


# In[631]:


px.bar(x=algortihms,y=scores)


# In[ ]:




