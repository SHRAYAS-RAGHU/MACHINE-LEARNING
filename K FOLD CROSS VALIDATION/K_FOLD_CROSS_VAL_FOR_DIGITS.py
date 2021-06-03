from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_digits
data = load_digits()
################################# CODING THE K-FOLD CROSS VALIDATION ##################################################333
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=3)

def get_score(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

sv = []
lr = []
rf = []

for tr_in, te_in in kf.split(data.data, data.target):
    x_train, x_test, y_train, y_test = data.data[tr_in], data.data[te_in], data.target[tr_in], data.target[te_in]
    sv.append(get_score(SVC(), x_train, x_test, y_train, y_test))
    lr.append(get_score(LogisticRegression(solver='liblinear',multi_class='ovr'), x_train, x_test, y_train, y_test))
    rf.append(get_score(RandomForestClassifier(n_estimators=40), x_train, x_test, y_train, y_test))
print('IMPLEMENTED K - FOLD',f"SVM : {sv}",f"LOGISTIC : {lr}",f"RANDOM FOREST CLASSIFICATION {rf}", sep = '\n')


"""
                                        USING THE INBUILT CROSS VALIDATION FUNCTION
                                                                                    """

from sklearn.model_selection import cross_val_score
print('\nINBUILT K - FOLD')
print('LOGISTIC',cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'), data.data, data.target, cv=3))
print('SVM',cross_val_score(SVC(), data.data, data.target, cv = 3))
print('RANDOM FOREST CLASSIFICATION', cross_val_score(
    RandomForestClassifier(n_estimators=40), data.data, data.target, cv=3))

"""
                                        CROSS VAL SCORE CAN ALS0 BE USED TO TEST SAME TYPE OF MODEL WITH DIFFERENT PARAMETERS FOR QUICK TUNING
                                                                                                                                            """