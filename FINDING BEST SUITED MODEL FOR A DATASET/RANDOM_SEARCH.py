import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC

iris_data = load_iris()
data = pd.DataFrame(iris_data.data, columns = iris_data.feature_names)
data['target'] = iris_data.target

x = iris_data.data
y = data.target

from sklearn.model_selection import RandomizedSearchCV
rscv = RandomizedSearchCV(
                            SVC(gamma = 'auto'),
                            {
                                'C':[1, 10, 20],
                                'kernel': ['rbf', 'linear']
                            },
                            cv = 5,
                            return_train_score= False,
                            n_iter=2,
                            random_state=9
                            )
rscv.fit(x, y)
result = pd.DataFrame(rscv.cv_results_)[['param_C', 'param_kernel', 'mean_test_score']]
print(result)