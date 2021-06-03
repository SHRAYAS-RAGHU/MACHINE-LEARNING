import pandas as pd
from sklearn.datasets import load_digits

digits_data = load_digits()

data = pd.DataFrame(digits_data.data)
data['target'] = digits_data.target

x = data.drop(['target'], axis='columns')
y = data.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, random_state = 10)

from sklearn.svm import SVC
model = SVC()
model.fit(x, y)

print(f"ACCURACY : {model.score(x_test, y_test) * 100}")

from sklearn.metrics import confusion_matrix
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

import matplotlib.pyplot as plt
import seaborn as sn
sn.heatmap(cm,annot = True)
plt.show()


