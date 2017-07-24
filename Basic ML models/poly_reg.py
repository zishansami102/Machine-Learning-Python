import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


theta = np.array([0,0,0])
alpha = 0.1
num_iter = 35

data = pd.read_csv('Philadelphia_Crime_Rate_noNA.csv')
data = np.array(data)
X = data[:, 1]
X -= np.mean(X)
X /= np.std(X)
X = np.vstack((np.ones(X.shape[0]), X, X**2)).T
y = data[:, 2]

m = X.shape[0]
for i in range(0, num_iter):
	h = X.dot(theta)
	theta = theta - (alpha)*((X.T).dot(h -y))/m

x = np.arange(-int(np.ceil(max(X[:,1])))/2,int(np.ceil(max(X[:,1]))),0.01)
x = np.vstack((np.ones(x.shape[0]), x, x**2)).T
regression_line = x.dot(theta)
plt.plot(x[:,1], regression_line, color='r')

plt.scatter(X[:,1],y, c='g')
plt.xlabel("House Prices in Philadelphia")
plt.ylabel("Crime Rate")
plt.show()
