import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

#Reading data
df = pd.read_csv('cancer.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
# df = df.astype(float).values.tolist()


X = np.array(df.drop(['class'],1, inplace=False))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)
print(acc)

testData = np.array([[8,7,5,10,7,9,5,5,4],[8,7,5,7,7,8,5,5,4]])
testData = testData.reshape(len(testData),-1)
pre = clf.predict(testData)

print(pre)
