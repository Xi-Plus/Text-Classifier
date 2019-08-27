import os
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


X = []
y = []

for infile in os.listdir('data/yes'):
    with(open(os.path.join('data/yes', infile), encoding='utf8')) as f:
        text = f.read()
        X.append(text)
        y.append(1)

for infile in os.listdir('data/no'):
    with(open(os.path.join('data/no', infile), encoding='utf8')) as f:
        text = f.read()
        X.append(text)
        y.append(0)

X_train, X_test, y_train, y_test = train_test_split(X, y)
print('train', len(y_train), 'test', len(y_test))
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(X_train)

classifier = MultinomialNB()
classifier.fit(counts, y_train)

example = vectorizer.transform(X_test)
predict = classifier.predict(example)
predict_proba = classifier.predict_proba(example)
# print(predict)
# print(np.array(y_test))

correct = 0
error1 = 0
error2 = 0
threshold = 0.9
for i in range(len(predict)):
    if predict_proba[i][1] > threshold:
        if y_test[i] == 1:
            correct += 1
        else:
            error2 += 1
            print('error2', predict_proba[i])
    else:
        if y_test[i] == 0:
            correct += 1
        else:
            error1 += 1
            print('error1', predict_proba[i])

print('-' * 50)
print('correct', correct, correct / len(y_test))
print('error1', error1, error1 / len(y_test))
print('error2', error2, error2 / len(y_test))
