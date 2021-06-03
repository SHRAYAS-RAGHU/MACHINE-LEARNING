import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model

data = pd.read_csv(r'C:\Users\krshr\Desktop\ML_CODING\LINEAR_REGRESSION\CANADA_PER_CAPITA.csv')
#print(data.head())
x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

reg = linear_model.LinearRegression()
#help(reg)
reg = reg.fit(x, y)

print(f"PREDICTION FOR THE YEAR 2020 IS {reg.predict([[2020]]).item()}")

plt.scatter(x, y)
plt.plot(x, reg.predict(x))
plt.show()