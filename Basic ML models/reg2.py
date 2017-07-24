import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

diabetes = datasets.load_diabetes()
X = diabetes.data[:, np.newaxis, 2]

X_train = X[:-20]
X_test = X[-20:]

y_train = diabetes.target[:-20]
y_test = diabetes.target[-20:]

# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)
clf = LinearRegression()
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test) * 100
# theta = clf.get_params()
print('Accuracy: ', acc)




plt.scatter(X_train, y_train,  color='red')
plt.scatter(X_test, y_test,  color='black')

plt.plot(X_train, clf.predict(X_train), color='red', linewidth=2)
plt.plot(X_test, clf.predict(X_test), color='black', linewidth=2)

# plt.xticks(())
# plt.yticks(())
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.show()

# print(df.head())