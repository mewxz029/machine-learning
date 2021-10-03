from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

dataset = pd.read_csv('NaiveBayesClassification/adult.csv')

# Transfrom data
def cleanData(dataset):
    for column in dataset.columns:
        if dataset[column].dtype == type(object):
            le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])
    return dataset

def split_feature_class(dataset, feature):
    features = dataset.drop(feature, axis=1)
    labels = dataset[feature].copy()
    return features, labels

dataset = cleanData(dataset)

# split train set and test set
train_set, test_set = train_test_split(dataset, test_size=0.2)

# split feature and label
train_features, train_labels = split_feature_class(train_set, "income")
test_features, test_labels = split_feature_class(test_set, "income")

# Model
model = GaussianNB()
model.fit(train_features, train_labels)

# Prediction
clf_pred = model.predict(test_features)

print("Accuracy = {:.2f}%".format(accuracy_score(test_labels, clf_pred) * 100))
print(pd.crosstab(test_labels, clf_pred, rownames=['Actually'], colnames=['Prediction']))