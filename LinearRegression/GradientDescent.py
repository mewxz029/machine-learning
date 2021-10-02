from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report


mnist_raw = loadmat("mnist-original.mat")
mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0]
}

# x = data, y = label
x, y = mnist["data"], mnist["target"]

# split train and test
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]


# print(mnist["data"].shape)
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# y_train = [0,0,.......,9.....,9]
predict_number = 5000
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier()
sgd_clf.fit(x_train, y_train_5)

y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3)
cm = confusion_matrix(y_train_5, y_train_pred)

y_test_pred = sgd_clf.predict(x_test)

classes = ['Other Number', 'Number 5']
print(classification_report(y_test_5, y_test_pred, target_names=classes))
