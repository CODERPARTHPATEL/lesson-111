import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('PokeMon Data.csv')
data.head()

data['Type 2'].fillna(value='None',inplace=True)
data.isnull().sum()

data['Type 1'].value_counts().plot.bar()
data['Type 2'].value_counts().plot.bar()
data['Legendary'].value_counts().plot.bar()

data['Type 1'].unique()
data['Type 2'].unique()

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

data['Legendary'] = lb.fit_transform(data['Legendary'])

data.head()

data.drop('Name',axis=1,inplace=True)

data = pd.get_dummies(data)

data.shape

from sklearn.model_selection import train_test_split
y=data.pop('Legendary')
x = data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(x_train,y_train)
ypred1 = model.predict(x_test)

from sklearn.metrics import accuracy_score
print('accurcy',accuracy_score(y_test,ypred1))

