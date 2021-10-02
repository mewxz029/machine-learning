from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

iris = load_iris()
x = iris['data']
y = iris['target']

x_train, x_test, y_train, y_test = train_test_split(x, y)

# Model
model = GaussianNB()
# Train
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy = {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))