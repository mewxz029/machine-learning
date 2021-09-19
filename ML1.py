import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5,5,10)
print(x)
y = 2*x+1

plt.plot(x, y, 'r', label='y = 2*x+1')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc="upper left")
plt.title("Graph Linear Regression")
plt.grid()
plt.show()