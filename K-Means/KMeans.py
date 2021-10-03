from operator import mod
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x, y = make_blobs(n_samples=300, centers=4, cluster_std=0.5, random_state=0)

# New point
x_test, y_test = make_blobs(n_samples=10, centers=4, cluster_std=0.5, random_state=0)

# Model
model = KMeans(n_clusters=4)
model.fit(x)
y_pred = model.predict(x)
y_pred_new = model.predict(x_test)
center = model.cluster_centers_

plt.scatter(x[:,0], x[:,1],c=y_pred)
plt.scatter(x_test[:,0], x_test[:,1], c=y_pred_new, s=120)
plt.scatter(center[0,0],center[0,1], label='Centroid 1')
plt.scatter(center[1,0],center[1,1], label='Centroid 2')
plt.scatter(center[2,0],center[2,1], label='Centroid 3')
plt.scatter(center[3,0],center[3,1], label='Centroid 4')
plt.legend()
plt.show()