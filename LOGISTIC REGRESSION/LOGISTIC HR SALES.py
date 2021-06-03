import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv(r'C:\Users\krshr\Desktop\ML_CODING\LOGISTIC REGRESSION\HR SALES.csv')

#print(data.groupby('left').mean())                             ########################## to explore reasons for employees leaving the job
#pd.crosstab(data.salary,data.left).plot(kind='bar')            ########################   PLOTTING THE CATEGORICAL VARIABLES
#plt.show()

"""         BY ANALYSIS OF GIVEN DATA WE FIND THAT 
            4 FEATURES ARE IMP 
                satisfaction_level
                average_montly_hours
                promotion_last_5years
                salary - categorical
"""
data = data[['satisfaction_level',
             'average_montly_hours', 'promotion_last_5years', 'salary', 'left']]

dummy_salary = pd.get_dummies(data['salary'], prefix = 'salary')
data = pd.concat([data, dummy_salary], axis = 'columns')
data = data.drop(['salary', 'salary_high'], axis = 'columns') ######### rule of thumb to drop one of the one hot encoded value
print(data.head())

x = data.drop(['left'], axis='columns')
y = data.left

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 8)
model = LogisticRegression()

model.fit(x_train, y_train)

print(model.score(x_test, y_test))

