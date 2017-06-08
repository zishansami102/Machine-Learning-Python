import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MeanShift
from sklearn import cross_validation, preprocessing

#	 Chnaging values from texts to numeric values
def textToNumerics(df):
	columns = df.columns.values
	for column in columns:
		textDict = {}
		def convert_to_int(val):
			return textDict[val]
		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
			column_contents = df[column].values.tolist()
			unique_elements = set(column_contents)
			x = 0
			for unique in unique_elements:
				if unique not in textDict:
					textDict[unique] = x
					x += 1
			df[column] = list(map(convert_to_int, df[column]))

	return df


df = pd.read_excel('titanic.xls')
print(df.head())
original_df = pd.DataFrame.copy(df)

df.drop(['name', 'body'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0,inplace=True)
df.drop(['home.dest','ticket'],1,inplace=True)
df = textToNumerics(df)

X = np.array(df.drop(['survived'],1,inplace=False))
X = preprocessing.scale(X)
clf = MeanShift()
# clf = KMeans(n_clusters=2)
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan

for i in range(len(X)):
	original_df['cluster_group'].iloc[i] = labels[i]

survival_rates = {}
n_clusters_ = len(np.unique(labels))
print(n_clusters_)
for i in range(n_clusters_):
	temp_df = original_df[(original_df['cluster_group']==float(i))]
	survival_cluster = temp_df[(temp_df['survived']==1)]
	survival_rate = float(len(survival_cluster))/len(temp_df)
	survival_rates[i] = survival_rate

print(survival_rates)

survived = np.array(df[:]['survived'])
correct = 0.0

# for i in range(len(labels)):
# 	if labels[i] == survived[i]:
# 		correct+=1

for i in range(len(X)):
	row = X[i,:]
	row = row.reshape(1,-1)
	predict = clf.predict(row)
	if predict == survived[i]:
		correct+=1

acc = correct/len(X)
print(df.head())
print(acc)


