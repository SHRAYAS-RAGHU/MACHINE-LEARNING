import pandas as pd
from sklearn.datasets import load_iris

iris_data = load_iris()
data = pd.DataFrame(iris_data.data)
data['target'] = iris_data.target

x = data.drop(['target'], axis = 'columns')
y = data.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 10)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

print(f"TEST_ACCURACY : {model.score(x_test, y_test)*100}")

from sklearn.metrics import confusion_matrix
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

################ FOR BETTER VISUALISATION USE SEABORN ###############3
import seaborn as sn
import matplotlib.pyplot as plt
#plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot= True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()