from scipy.io import loadmat
import matplotlib.pyplot as plt

mnist_raw = loadmat("mnist-original.mat")
mnist = {
    # Row -> Column Swap
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0]
}

# x = data, y = label
x, y = mnist["data"], mnist["target"]

number = x[35000]
number_image = number.reshape(28, 28)

print(y[35000])
plt.imshow(
    number_image, 
    cmap=plt.cm.viridis, 
    interpolation="sinc"
)
plt.show()
