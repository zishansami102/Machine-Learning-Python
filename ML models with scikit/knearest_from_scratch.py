import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import random

# dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,7],[7,8],[8,6]]}
# testD = [0,0]

def euclDist(p, q):
	p = np.array(p)
	q = np.array(q)
	dist = np.sqrt(sum((p-q)**2))
	return dist


def knearest(data, point, k=3):
	if len(data)>=k:
		return "err : Dataset provided do not have enough points"
	distances = []
	for i in data:
		for ii in data[i]:
			dist = euclDist(ii, point)
			distances.append([dist, i])
	votes = [i[1] for i in sorted(distances)[0:k]]
	votes_results = Counter(votes).most_common(1)[0][0]

	return votes_results



df = pd.read_csv('cancer.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'],1,inplace=True)
df = df.astype(float).values.tolist()
random.shuffle(df)

df_train = df[:-int(0.1*len(df))]
df_test = df[-int(0.1*len(df)):]

train_data = {2:[], 4:[]}
for i in df_train:
	train_data[i[-1]].append(i[:-1])
test_data = {2:[], 4:[]}
for i in df_test:
	test_data[i[-1]].append(i[:-1])

count = 0.0
total = 0
for group in test_data:
	for point in test_data[group]:
		prediction = knearest(train_data, point, k=5)
		if prediction==group:
			count += 1
		total +=1

acc = count/total
print(acc)
