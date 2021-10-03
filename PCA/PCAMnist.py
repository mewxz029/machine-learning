from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.io import loadmat
import matplotlib.pyplot as plt

mnist_raw = loadmat("mnist-original.mat")
mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0]
}

x_train, x_test, y_train, y_test = train_test_split(mnist["data"], mnist["target"], random_state=0)

pca = PCA(.95)
data = pca.fit_transform(x_train)
result = pca.inverse_transform(data)

# Show image
plt.figure(figsize=(8, 4))
# Image feature 784
plt.subplot(1,2,1)
plt.imshow(x_train[5500].reshape(28, 28), cmap=plt.cm.gray, interpolation='nearest')
plt.xlabel("784 Feature")
plt.title("Original")

# Image feature 154
plt.subplot(1,2,2)
plt.imshow(result[5500].reshape(28, 28), cmap=plt.cm.gray, interpolation='nearest')
plt.xlabel("{} Feature".format(pca.n_components_))
plt.title("PCA Image")
plt.show()

