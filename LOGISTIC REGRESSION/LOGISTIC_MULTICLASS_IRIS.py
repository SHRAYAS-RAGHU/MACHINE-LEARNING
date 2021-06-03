import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix

data = load_iris()
print(dir(data))
print(data.data[:5], data.target[:5])

x = data.data
y = data.target

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 8)
model = LogisticRegression()

model.fit(x_train, y_train)

print('ACCURACY :', model.score(x_test, y_test),
      f"\nOUTPUT FOR [5.1 3.5 1.4 0.2] {data.target_names[model.predict([[5.1, 3.5, 1.4, 0.2]])]}")

pred = model.predict(x_test)
conf_mat = confusion_matrix(y_test, pred)
print(conf_mat)

sn.heatmap(conf_mat, annot=True)
plt.show()