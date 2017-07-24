import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = np.array(pd.read_csv('Philadelphia_Crime_Rate_noNA.csv'))
X = (data[:, 1] - np.mean(data[:, 1]))/np.std(data[:, 1])
X = np.vstack((np.ones(X.shape[0]), X, X**2)).T

theta = np.array([0,0,0])
for i in range(0, 100):
	theta = theta - (0.1)*((X.T).dot(X.dot(theta) -data[:, 2]))/X.shape[0]

x = np.arange(-int(np.ceil(max(X[:,1])))/1.5,int(np.ceil(max(X[:,1])))*1.5,0.01)
x = np.vstack((np.ones(x.shape[0]), x, x**2)).T
plt.plot(x[:,1], x.dot(theta), color='r')
plt.scatter(X[:,1],data[:,2], c='g')
plt.xlabel("House Prices in Philadelphia")
plt.ylabel("Crime Rate")
plt.show()
