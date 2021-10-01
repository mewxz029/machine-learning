from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier


def displayImage(x):
    plt.imshow(
        x.reshape(28, 28),
        cmap = plt.cm.binary,
        interpolation = "nearest"
    )
    plt.show()

def displayPredict(clf, actually_y, x):
    print("Actually = ", actually_y)
    print("Prediction = ", clf.predict([x])[0])

def Classifier(target_number, pred_number, x_train, x_test, y_train ,y_test):
    y_train_target = (y_train == target_number)
    y_test_target = (y_test == target_number)

    sgd_clf = SGDClassifier()
    sgd_clf.fit(x_train, y_train_target)

    displayPredict(sgd_clf, y_test_target[pred_number], x_test[pred_number])
    displayImage(x_test[pred_number])

mnist_raw = loadmat("mnist-original.mat")
mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0]
}

x, y = mnist["data"], mnist["target"]
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

Classifier(0, 5000, x_train, x_test, y_train, y_test)

