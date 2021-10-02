import seaborn as sb
import matplotlib.pyplot as plt
iris_dataset = sb.load_dataset('iris')

sb.set()
sb.pairplot(iris_dataset, hue='species', size=2)
plt.show()