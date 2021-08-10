#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[53]:


data_train = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header = None)
data_test = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', skiprows = 1, header = None)


# In[54]:


data_train.tail(3)


# In[55]:


data_test.tail(3)


# # Adding colum to dataset

# In[56]:


column = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 
                'occupation','relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
                 'native_country', 'wage_class']
data_train.columns = column
data_test.columns = column


# In[57]:


data_train.head(3)


# In[58]:


df = pd.concat([data_train,data_test])
df['workclass'].value_counts()


# In[59]:


for column in df.columns:
    print(f" value count for {column} : \n {df[column].value_counts()}")


# df.info()

# In[60]:


df.describe()


# # replacing ? from workclass column

# In[61]:


df.replace('?',np.nan,inplace=True)


# In[62]:


df.wage_class.unique()


# In[66]:


df.replace({' <=50K':0,' >50K':1,' <=50K.':0,' >50K.':1}).head(3)


# In[67]:


df['workclass'].fillna('0',inplace=True)


# In[69]:


plt.figure(figsize=(10,8))
sns.countplot(df['workclass'])
plt.xticks(rotation = 45)
plt.show()


# In[70]:


df['education'].value_counts()


# In[71]:


df.columns


# In[73]:


sns.catplot(x='education',y='wage_class',data=df,height=10,palette='muted',kind='bar')
plt.xticks(rotation=45)
plt.show()


# In[74]:


def primary(x):
    if x in [' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' 10th', ' 11th', ' 12th']:
        return 'Primary'
    else:
        return x


# In[75]:


df['education'] = df['education'].apply(primary)


# In[76]:


sns.catplot(x='education',y='wage_class',data=df,height=10,palette='muted',kind='bar')
plt.xticks(rotation=45)
plt.show()


# In[77]:


df['marital_status'].replace(' Married-AF-spouse', ' Married-civ-spouse',inplace=True)


# In[78]:


sns.catplot(x='marital_status',y='wage_class',data=df,palette='muted',kind='bar',height=8)
plt.xticks(rotation=45)
plt.show()


# In[79]:


df['occupation'].fillna('0',inplace=True)
df['occupation'].value_counts()
df['occupation'].replace(' Armed-Forces','0',inplace=True)
df['occupation'].value_counts()
sns.catplot(x='occupation',y='wage_class',data=df,palette='muted',kind='bar',height=8)
plt.xticks(rotation=45)


# In[81]:


df['relationship'].value_counts()


# In[82]:


df['race'].value_counts()


# In[83]:


df.columns


# In[84]:


df['sex'].value_counts()


# In[85]:


df['native_country'].unique()


# In[86]:


def native(country):
    if country in [' United-States',' Canada']:
        return 'North_America'
    elif country in [' Puerto-Rico',' El-Salvador',' Cuba',' Jamaica',' Dominican-Republic',' Guatemala',' Haiti',' Nicaragua',' Trinadad&Tobago',' Honduras']:
        return 'Central_America' 
    elif country in [' Mexico',' Columbia',' Vietnam',' Peru',' Ecuador',' South',' Outlying-US(Guam-USVI-etc)']:
        return 'South_America'
    elif country in [' Germany',' England',' Italy',' Poland',' Portugal',' Greece',' Yugoslavia',' France',' Ireland',' Scotland',' Hungary',' Holand-Netherlands']:
        return 'EU'
    elif country in [' India',' Iran',' China',' Japan',' Thailand',' Hong',' Cambodia',' Laos',' Philippines',' Taiwan']:
        return 'Asian'
    else:
        return country


# In[88]:


df['native_country'] = df['native_country'].apply(native)


# In[89]:


sns.catplot(x='native_country',y='wage_class',data=df,palette='muted',kind='bar',height=8)
plt.xticks(rotation=45)


# In[90]:


corr = df.corr()
plt.figure(figsize=(10,12))
sns.heatmap(corr,annot=True)


# In[91]:


X = df.drop(['wage_class'],axis=1)
y = df['wage_class']


# In[92]:


X.columns


# In[94]:


X_dummy = pd.get_dummies(X)


# In[95]:


X_dummy.head()


# In[96]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_d)


# In[97]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.3,random_state=101)


# In[98]:


parameters = [{ 'learning_rate':[0.01,0.001],
                        'max_depth': [3,5,10],
                        'n_estimators':[10,50,100,200]
                    }
                   ]


# In[102]:


from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
Xbc = XGBClassifier()
Grid_cv = GridSearchCV(Xbc,parameters,scoring='accuracy',cv=5,n_jobs=3,verbose=3)
Grid_cv.fit(x_train,y_train)


# In[104]:


Grid_cv.best_params_


# In[105]:


XBC = XGBClassifier(learning_rate=0.01,max_depth=10,n_estimators=200)
XBC.fit(x_train,y_train)


# In[106]:


XBC.score(x_test,y_test)


# In[108]:


y_pred = XBC.predict(x_test)


# In[109]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[112]:


print(f'Accuracy Score:{accuracy_score(y_test,y_pred)}')
print(f'Confusion Matrix:{confusion_matrix(y_test,y_pred)}')
print(f'Classification Report: {classification_report(y_test,y_pred)}')


# In[ ]:




