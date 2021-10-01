from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

def displayImage(x):
    plt.imshow(
        x.reshape(28, 28),
        cmap=plt.cm.viridis, 
        interpolation="sinc"
    )
    plt.show()

def displayPredict(clf, actually_y, x, score):
    print("Actually = ", actually_y)
    print("Prediction = ", clf.predict([x])[0])
    print("Accuracy: {0:0.2f}% (+/- {1:0.2f})".format(score.mean() * 100, score.std() * 2))

def Classifier(target_number, pred_number, x_train, x_test, y_train ,y_test):
    y_train_target = (y_train == target_number)
    y_test_target = (y_test == target_number)

    sgd_clf = SGDClassifier()
    sgd_clf.fit(x_train, y_train_target)

    score = cross_val_score(sgd_clf, x_train, y_train_target, cv= 3, scoring="accuracy")
    
    displayPredict(sgd_clf, y_test_target[pred_number], x_test[pred_number], score)
    displayImage(x_test[pred_number])

mnist_raw = loadmat("mnist-original.mat")
mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0]
}

x, y = mnist["data"], mnist["target"]
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

Classifier(0, 5000, x_train, x_test, y_train, y_test)