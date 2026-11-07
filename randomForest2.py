import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix


iris = load_iris()
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df['species']= iris.target

print(df.head())

x = df.drop('species',axis=1)
y =df['species']

x_test,x_train,y_test,y_train = train_test_split(
    x,y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print('\nAccuracy',accuracy_score(y_test,y_pred))

print('\nClassification report')
print(classification_report(y_test,y_pred))

print('\nconfusion matrix')
print(confusion_matrix(y_test,y_pred))


importance = pd.DataFrame({
    'Feature':x.columns,
    'Importance':model.feature_importances_
})

print('\mFeture imporance')
print(importance.sort_values(by='Importance',ascending=False))