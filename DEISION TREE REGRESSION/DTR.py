import numpy as np
import pandas as pd
from sklearn import tree

data = pd.read_csv(r'C:\Users\krshr\Desktop\ML_CODING\DEISION TREE REGRESSION\TITANIC.csv')
data = data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
#print(data.head())
#print(data.isnull().sum())   ################### find the values which are missing

med = data.Age.median()
data.Age = data.Age.fillna(med)  ############### fill the values with medians
#print(data.isnull().sum())
"""
from sklearn.preprocessing import LabelEncoder
le_sex = LabelEncoder()
data['gender'] = le_sex.fit_transform(data['Sex'])
data = data.drop(['Sex'], axis = 'columns')
"""
dum = pd.get_dummies(data['Sex'], prefix='gender')
data = pd.concat([data, dum], axis = 'columns')
data = data.drop(['Sex', 'gender_male'], axis = 'columns')

x = data.drop(['Survived'], axis = 'columns')
y = data.Survived

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 10)

model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)
print('ACCURACY :', model.score(x_test, y_test) * 100)