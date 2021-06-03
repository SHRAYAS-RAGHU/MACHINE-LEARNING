import pandas as pd
""" IF FEATURES BINARY BINOMIALNB IF RANGE OF INTEGERS THEN MULTINOMIALNB ELSE GUASSIAN IF CONTINUOUS"""
from sklearn.naive_bayes import MultinomialNB     

data = pd.read_csv(r'C:\Users\krshr\Desktop\ML_CODING\NAIVE_BASE\EMAIL_SPAM.csv')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Category'] = le.fit_transform(data['Category'])
#print(data.head())

#print(data.groupby('Category').describe())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.Message, data.Category)

"""                                                        WITHOUT USING PIPLINE   """
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()                                #  ASSIGNING VALUES TO TEXTS BASED ON COUNT
text_count = v.fit_transform(X_train.values)

model = MultinomialNB()
model.fit(text_count, y_train)

X_test_count = v.transform(X_test)
print('WITHOUT PIPELINE',model.score(X_test_count, y_test)*100)

########################### ONE DISADVANTAGE IS WE HAVE TO TRANSFORM EVERYTIME TO AVOID IT WE CAN USE PIPELINE ################################3
from sklearn.pipeline import Pipeline
new_model = Pipeline([
                ('VECTORIZER', CountVectorizer()),
                ('NAIVE_BASE', MultinomialNB())
                ])
new_model.fit(X_train, y_train)

print('WITH PIPELINE',new_model.score(X_test, y_test)*100)

######################## TESTING THE EMAIL ####################

emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
spam = {0:'NOT SPAM', 1:'SPAM'}
for test_data, prediction in zip(emails, new_model.predict(emails)):
    print(test_data, ':',spam[prediction])

