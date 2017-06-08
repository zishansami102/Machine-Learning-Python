import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans

X = np.array([[1,2],
			[1.5,1.8],
			[5,8],
			[8,8],
			[1,0.6],
			[9,11]])

# plt.scatter(X[:,0], X[:,1], s=150)
# plt.show()

clf = KMeans(n_clusters=2)
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

colors = ['r', 'g']

for i in range(len(X)):
	plt.scatter(X[i][0],X[i][1], color=colors[labels[i]])
plt.scatter(centroids[:,0], centroids[:,1], marker='x', color='b')
plt.show()
