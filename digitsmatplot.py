import matplotlib.pyplot as plt
from sklearn import datasets

digits_dataset = datasets.load_digits()
plt.imshow(digits_dataset.images[0], cmap=plt.get_cmap('gray'))
plt.show()