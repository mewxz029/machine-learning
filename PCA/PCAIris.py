from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import seaborn as sb
from sklearn.decomposition import PCA

iris = sb.load_dataset('iris')
x = iris.drop('species', axis=1)
y = iris['species']

# PCA
pca = PCA(n_components=3)
x_pca = pca.fit_transform(x)

x['PCA1'] = x_pca[:,0]
x['PCA2'] = x_pca[:,1]
x['PCA3'] = x_pca[:,2]

x_train, x_test, y_train, y_test = train_test_split(x, y)

# Complete Data
x_train = x_train.loc[:, ['PCA1', 'PCA2', 'PCA3']]
x_test = x_test.loc[:, ['PCA1', 'PCA2', 'PCA3']]

# Model
model = GaussianNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy : {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))