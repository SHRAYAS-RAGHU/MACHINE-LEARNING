import pandas as pd
import numpy as np

data = pd.read_csv(r'C:\Users\krshr\Desktop\ML_CODING\METRICS\metrics.csv')
x = data['Predicted Class']
y = data['Actual Class']

######################################### CONFUSION MATRIX, PRECISION, RECALL ANF F1 SCORE ###############################################

################################################# FROM SCRATCH ############################################################
cm = {}
for i in range(4):
    for j in range(4):
        cm[(i,j)] = 0

for i in range(len(x)):
    cm[(y[i], x[i])] += 1

cm = np.array([i for i in cm.values()])
cm = cm.reshape((4,4))
tp = cm[0, 0]
tn = np.sum(cm[1: , 1:])
fp = np.sum(cm[1:, 0])
fn = np.sum(cm[0, 1:])

recall = tp / (tp + fn)
precision = tp / (tp + fp)
f1 = 2 * (precision *recall) / (precision + recall)

print('CONFUSION MATRIX :\n',cm)
print(f'\nCLASS 0 : \n TRUE POSITIVE : {tp} \n TRUE NEGATIVE : {tn} \n FALSE POSITIVE : {fp} \n FALSE NEGATIVE : {fn} \n')
print('FROM SCRATCH : \n')
print(f'RECALL : {recall} \nPRECISION {precision} \nF1 - SCORE : {f1}')

##################################### USING INBUILT METHODS OF SKLEARN ##########################################################

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, x)
import matplotlib.pyplot as plt
import seaborn as sn
import sklearn.metrics as sm
print('\nUSING SKLEARN : \n')
print('RECALL :',sm.recall_score(y,x, average='macro', labels=[0]))
print('PRECISION :', sm.precision_score(y, x, average='macro', labels=[0]))
print('F1 SCORE :', sm.f1_score(y, x, average='macro', labels=[0]))

sn.heatmap(cm, annot=True)
plt.show()
