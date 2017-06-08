import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

input = np.loadtxt("ex1data1.txt", dtype='i', delimiter=',')

X = input[:,0]
y = input[:,1]
X = X.reshape((len(X),1))
y = y.reshape((len(y),1))
print(X.shape, y.shape)
mod = LinearRegression()
mod.fit(X, y)


plt.scatter(X, y, color='black')
plt.xlabel('Population of city in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X, mod.predict(X), color='red', linewidth='2')
plt.show()


