import numpy as np
import matplotlib.pyplot as plt

# create Array number sample space => -5 to 5 length = 10
x = np.linspace(-5,5,10)
print(x)
y = 2*x+1

plt.scatter(x, y, color='black')
plt.plot(x, y, 'r', label='y = 2*x+1')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc="upper left")
plt.title("Graph Linear Regression")
plt.grid()
plt.show()