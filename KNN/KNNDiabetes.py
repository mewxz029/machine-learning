from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("KNN/diabetes.csv")

# Prepare Data
x = df.drop("Outcome", axis=1).values
y = df['Outcome'].values

# Split Train set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

knn = KNeighborsClassifier(n_neighbors=8)
# Training
knn.fit(x_train, y_train)

# Prediction
y_pred = knn.predict(x_test)

# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
print(pd.crosstab(y_test, y_pred, rownames=['Actually'], colnames=['Prediction'], margins=True))
print("Accuracy = {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

# Find K to Model for Most Accuracy
# k_neighbors = np.arange(1,9)
# # Empty Array
# train_score = np.empty(len(k_neighbors))
# test_score = np.empty(len(k_neighbors))

# for i, k in enumerate(k_neighbors):
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(x_train, y_train)
#     # Test Efficient
#     train_score[i] = knn.score(x_train, y_train)
#     test_score[i] = knn.score(x_test, y_test)

#     print(test_score[i] * 100)

# plt.title("Compare K in Model")
# plt.plot(k_neighbors, test_score, label='Test Score')
# plt.plot(k_neighbors, train_score, label='Train Score')
# plt.legend()
# plt.xlabel("K Number")
# plt.ylabel("Score")
# plt.show()
