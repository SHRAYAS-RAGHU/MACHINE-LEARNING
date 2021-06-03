import pandas as pd
from sklearn.datasets import load_iris

iris_data = load_iris()
data = pd.DataFrame(iris_data.data, columns = iris_data.feature_names)
data['target'] = iris_data.target

x = iris_data.data
y = data.target

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

model_params = {
                'svm' : {
                        'model' :SVC(gamma = 'auto'),
                        'params': {
                                    'C':[1, 10, 20],
                                    'kernel': ['rbf','linear']
                                    }
                    	},
                'random_forest' : {
                        'model': RandomForestClassifier(),
                        'params': {
                                'n_estimators' : [1, 5, 10]
                                    }
                                },
                'logistic_regression': {
                        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
                        'params': {
                            'C': [1, 5, 10]
                                  }
                                }
                }

score = []

from sklearn.model_selection import GridSearchCV

score = []

for model_name, mp in model_params.items():
    gcv = GridSearchCV(mp['model'], mp['params'],
                       cv=5, return_train_score=False)
    gcv.fit(x, y)
    score.append({
        'model': model_name,
        'best_score': gcv.best_score_,
        'best_params': gcv.best_params_
    })

score = pd.DataFrame(score, columns=['model','best_score','best_params'])
print(score)
"""
                                                             model  best_score                best_params
                                            0                  svm    0.980000  {'C': 1, 'kernel': 'rbf'}
                                            1        random_forest    0.966667        {'n_estimators': 1}
                                            2  logistic_regression    0.966667                   {'C': 5}
"""
