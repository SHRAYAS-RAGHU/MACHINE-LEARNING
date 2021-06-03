import pandas as pd
import matplotlib.pyplot as plt
from word2number import w2n                 # TO CONVERT WORD TO NOS. LIKE 'ONE' : 1, 'ELEVEN' : 11
from sklearn import linear_model

data = pd.read_csv(r'C:\Users\krshr\Desktop\ML_CODING\LINEAR_REGRESSION\EMPLOYEE_RECRUITMENT.csv')

data.experience = data.fillna('zero')                    
data.experience = data.experience.apply(w2n.word_to_num)

fill_value = data['test_score(out of 10)'].median()
data['test_score(out of 10)'] = data['test_score(out of 10)'].fillna(fill_value)

x = data.iloc[:, :-1].values
y = data.iloc[:, 3:].values                                  ########## data.iloc[:, -1] gives values as a single dim array here 2d req

reg = linear_model.LinearRegression()
reg = reg.fit(x, y)

print(f"PREDICTION FOR APPLICANT WITH 2YRS EXP TEST SCORE OF 9 INTERVIEW SCORE OF 6 {reg.predict([[2,9,6]]).item()}")
print(f"WEIGHTS {reg.coef_} BIAS {reg.intercept_}")

import pickle
with open('model_pickle', 'wb') as f:   ############ wb for write binary
    pickle.dump(reg,f)
f.close() 
with open('model_pickle', 'rb') as f:  ############# rb for read binary 
    m = pickle.load(f)
print('LOADED MODEL PREDICTION',m.predict([[2,9,6]]).item())
f.close()
