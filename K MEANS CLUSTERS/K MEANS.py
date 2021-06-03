import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv(r'C:\Users\krshr\Desktop\ML_CODING\K MEANS CLUSTERS\INCOMES.csv')

x = data.drop(['Name'], axis = 'columns')

"""                                          IF DATA IS NOT SCALED WHILE OPERATING WITH DIFFERENT FEATURES THERE MAY BE ERRONOUS CLUSTERING   """

scaler = MinMaxScaler()

x[['Income($)']] = scaler.fit_transform(x[['Income($)']])  ##### FIT - fits the data into the scaler and transform transforms the data

x[['Age']] = scaler.fit_transform(x[['Age']])

print(x.head())
km = KMeans(n_clusters=3)
y_pred = km.fit_predict(x)

data['scatter'] = y_pred
df0 = data[data['scatter'] == 0]
df1 = data[data['scatter'] == 1]
df2 = data[data['scatter'] == 2]

import matplotlib.pyplot as plt
plt.scatter(df0['Age'], df0['Income($)'])
plt.scatter(df1['Age'], df1['Income($)'])
plt.scatter(df2['Age'], df2['Income($)'])
plt.show()

"""                          WHEN THERE ARE LIMITED NO. OF FEATURES IT CAN BE PLOTTED BUT NOT FOR HIGHER DIM OF FEATURES  
                                                    USE ELBOW PLOT OF SSE TO FIND BEST FIT K """

sse = []
for i in range(1, 11):
    km = KMeans(n_clusters=i)
    km.fit(x)
    sse.append(km.inertia_)
plt.plot(range(1,11), sse)
plt.show()