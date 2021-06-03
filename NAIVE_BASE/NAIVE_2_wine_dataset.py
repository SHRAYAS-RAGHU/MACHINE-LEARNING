import pandas as pd
""" IF FEATURES BINARY, BINOMIALNB IF RANGE OF INTEGERS THEN MULTINOMIALNB ELSE GUASSIAN IF CONTINUOUS"""    
from sklearn.datasets import load_wine 

wine_data = load_wine()
#print(dir(wine_data), wine_data.feature_names, wine_data.target_names)
data = pd.DataFrame(wine_data.data, columns = wine_data.feature_names)
data['Target'] = wine_data.target
#print(data.head())

x = data.drop(['Target'], axis = 'columns')
y = data.Target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 10)

from sklearn.naive_bayes import MultinomialNB, GaussianNB

model_1 = MultinomialNB()
model_2 = GaussianNB()

model_1.fit(X_train, y_train)
model_2.fit(X_train, y_train)

print(f"ACCURACY MULTINOMINAL: {model_1.score(X_test, y_test)*100:.4f} ACCURACY GAUSSIAN {model_2.score(X_test, y_test)*100:.4f}")

