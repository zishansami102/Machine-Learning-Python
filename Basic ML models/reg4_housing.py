import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle

input = np.loadtxt("ex1data2.txt", dtype='i', delimiter=',')
print(input.shape)
X = input[0:-20,0:2]
y = input[0:-20,2]
X_test = input[-20:,0:2]
y_test = input[-20:,2]

mod = LinearRegression(n_jobs=-1)
mod.fit(X, y)

# for saving the model
with open('housing.pickle', 'wb') as f:
	pickle.dump(mod, f)

# # for fetching the saved model
# pickle_in = open('housing.pickle', 'rb')
# mod = pickle.load(pickle_in)


acc = mod.score(X_test, y_test)

print(acc)
print(mod.predict([2104, 4]))



