import pandas as pd
from sklearn.naive_bayes import GaussianNB     ###### NAIVE BCOZ VARIABLES ARE CONSIDERED INDEPENDENT

data = pd.read_csv(r'C:\Users\krshr\Desktop\ML_CODING\NAIVE_BASE\TITANIC.csv')
data = data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
#print(data.head())

dum = pd.get_dummies(data['Sex'])
data = pd.concat([data, dum], axis='columns')
data = data.drop(['Sex'], axis ='columns')


data.Age = data.Age.fillna(data.Age.mean()) ### FILLING NAN VALUES

x = data.drop(['Survived'], axis='columns')
y = data.Survived

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

model = GaussianNB()
model.fit(x_train, y_train)

print(model.score(x_test, y_test) * 100)

from sklearn.model_selection import cross_val_score
print(cross_val_score(GaussianNB(),x_train, y_train, cv=5))
