import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC

iris_data = load_iris()
data = pd.DataFrame(iris_data.data, columns = iris_data.feature_names)
data['target'] = iris_data.target

x = iris_data.data
y = data.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, data.target, test_size=0.3, random_state = 1)

from sklearn.model_selection import GridSearchCV
gcv = GridSearchCV(
        SVC(gamma='auto'),
        {
            'C':[1,10,20],
            'kernel' : ['rbf', 'linear']    
        }, cv = 5, return_train_score=False
        )

gcv.fit(X_train, y_train)
result_frame = pd.DataFrame(gcv.cv_results_)
print(result_frame[['param_C', 'param_kernel', 'mean_test_score']]) 
print(f"BEST PARAMETERS : {gcv.best_params_} BEST SCORE FOR THE MODEL {gcv.best_score_}")
