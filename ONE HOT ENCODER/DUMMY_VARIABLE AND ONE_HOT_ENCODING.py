import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.index_tricks import AxisConcatenator
import pandas as pd
################################ IF U WANT TO AVOID DUMMY VARIABLE TRAP, IF THERE R 3 DUMMIES DROP ONE OF THE DUMMY COLUMNS ##########################
data = pd.read_csv(r'C:\Users\krshr\Desktop\ML_CODING\ONE HOT ENCODER\CAR PRICES.csv')
dummies = pd.get_dummies(data['Car Model'])
data = pd.concat([data, dummies], axis='columns')
data = data.drop(['Car Model','Mercedez Benz C class'], axis='columns')

x = data.drop(['Sell Price($)'], axis = 'columns')
y = data['Sell Price($)']

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x,y)

print(f"AFTER DATA CLEANSING LINEAR MODEL PREDICTION FOR [69000, 6, 0, 1] IS {reg.predict([[69000, 6, 0, 1]]).item()}")
print('ACCURACY : ',reg.score(x, y) * 100)