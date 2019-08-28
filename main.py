import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


class TextClassifier():
    def __init__(self, trainall=False):
        self.texts = {}
        self.X = []
        self.y = []
        self.vectorizer = None
        self.classifier = None

        BASEDIR = os.path.dirname(os.path.abspath(__file__))

        for infile in os.listdir(os.path.join(BASEDIR, 'data/yes')):
            with(open(os.path.join(BASEDIR, 'data/yes', infile), encoding='utf8')) as f:
                filename = infile.replace('.txt', '')
                text = f.read()
                self.X.append(text)
                self.y.append(1)
                self.texts[filename] = text

        for infile in os.listdir(os.path.join(BASEDIR, 'data/no')):
            with(open(os.path.join(BASEDIR, 'data/no', infile), encoding='utf8')) as f:
                filename = infile.replace('.txt', '')
                text = f.read()
                self.X.append(text)
                self.y.append(0)
                self.texts[filename] = text

        self.vectorizer = CountVectorizer()
        self.classifier = MultinomialNB()
        X_train, self.X_test, y_train, self.y_test = train_test_split(self.X, self.y)
        if trainall:
            print('train', len(self.y))
            counts = self.vectorizer.fit_transform(self.X)
            self.classifier.fit(counts, self.y)
        else:
            print('train', len(y_train), 'test', len(self.y_test))
            counts = self.vectorizer.fit_transform(X_train)
            self.classifier.fit(counts, y_train)

    def predict(self, text, ans):
        example = self.vectorizer.transform(text)
        # predict = self.classifier.predict(example)
        predict_proba = self.classifier.predict_proba(example)
        # print(predict)
        # print(np.array(y_test))
        return predict_proba

    def print(self, predict_proba, ans):
        correct = 0
        error1 = 0
        error2 = 0
        threshold = 0.9
        for i in range(len(predict_proba)):
            if predict_proba[i][1] > threshold:
                if ans[i] == 1:
                    correct += 1
                else:
                    error2 += 1
                    print('error2', predict_proba[i])
            else:
                if ans[i] == 0:
                    correct += 1
                else:
                    error1 += 1
                    print('error1', predict_proba[i])

        print('-' * 50)
        print('correct', correct, correct / len(ans))
        print('error1', error1, error1 / len(ans))
        print('error2', error2, error2 / len(ans))


if __name__ == "__main__":
    classifier = TextClassifier()
    res = classifier.predict(classifier.X_test, classifier.y_test)
    classifier.print(res, classifier.y_test)
