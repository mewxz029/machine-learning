import pylab
from sklearn import datasets
digits_dataset = datasets.load_digits()
print(digits_dataset.keys())
print(digits_dataset.target[:10])
pylab.imshow(digits_dataset.images[1], cmap = pylab.cm.gray_r)
pylab.show()
