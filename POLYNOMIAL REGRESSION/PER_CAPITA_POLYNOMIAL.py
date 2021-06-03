import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

data = pd.read_csv(r'C:\Users\krshr\Desktop\ML_CODING\LINEAR_REGRESSION\CANADA_PER_CAPITA.csv')
#print(data.head())
x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

reg = linear_model.LinearRegression()
#help(reg)
P = PolynomialFeatures(degree=4)
x_p = P.fit_transform(x)
#print(x[0], x_p[0])
reg = reg.fit(x_p, y)

print(f"PREDICTION FOR THE YEAR 2020 IS {reg.predict(P.fit_transform([[2020]])).item()}")

plt.scatter(x, y)
plt.plot(x, reg.predict(x_p))
plt.show()