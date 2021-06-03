from sklearn.model_selection import train_test_split
import pandas as pd
data = pd.read_csv(r'C:\Users\krshr\Desktop\ML_CODING\TRAIN TEST SPLIT\CARPRICES.csv')
x = data.drop(['Sell Price($)'], axis= 'columns')
y = data['Sell Price($)']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=10)
print(f"TOTAL LENGTH : {len(x)} X_TRAIN {len(x_train)} X_TEST {len(x_test)} Y_TRAIN {len(y_train)} Y_TEST {len(y_test)}")
