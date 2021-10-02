from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import itertools

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

def displayClassificationReport(target_number, y_test_target, y_test_pred):
    classes = ['Other Number', 'Number {:d}'.format(target_number)]
    print(classification_report(y_test_target, y_test_pred, target_names=classes))
    print("Accuracy Score = {:0.2f}%".format(accuracy_score(y_test_target, y_test_pred)*100))

def displayConfusionMatrix(cm, cmap=plt.cm.GnBu):
    classes = ["Other Number", "Number 5"]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    trick_marks = np.arange(len(classes))
    plt.xticks(trick_marks, classes)
    plt.yticks(trick_marks, classes)
    thresh = cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
        horizontalalignment = 'center',
        color='white' if cm[i, j] > thresh else 'black' )
    
    plt.tight_layout()
    plt.ylabel('Actually')
    plt.xlabel('Prediction')
    plt.show()

def classifier(target_number, pred_number, x_train, x_test, y_train ,y_test):
    y_train_target = (y_train == target_number)
    y_test_target = (y_test == target_number)

    sgd_clf = SGDClassifier()
    sgd_clf.fit(x_train, y_train_target)

    # score = cross_val_score(sgd_clf, x_train, y_train_target, cv=3, scoring="accuracy")
    
    y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_target, cv=3)
    # cm = confusion_matrix(y_train_target, y_train_pred)

    y_test_pred = sgd_clf.predict(x_test)
    
    displayClassificationReport(target_number, y_test_target, y_test_pred)
    # displayPredict(sgd_clf, y_test_target[pred_number], x_test[pred_number], score)
    # plt.figure()
    # displayConfusionMatrix(cm)
    # displayImage(x_test[pred_number])

mnist_raw = loadmat("mnist-original.mat")
mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0]
}

x, y = mnist["data"], mnist["target"]
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

classifier(5, 5000, x_train, x_test, y_train, y_test)