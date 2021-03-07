import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv('Iris.csv')

predictors = data.iloc[:,1:5]
target = data.iloc[:,-1]

X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.3, stratify=target)

metr='euclidean' #other options:{minkowski','manhattan'}
K=15
knnmodel = KNeighborsClassifier(n_neighbors=K, metric=metr)
knnmodel.fit(X_train, Y_train)
Y_predict = knnmodel.predict(X_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(Y_test, Y_predict)

pickle.dump(knnmodel,open('knnmodel.pkl','wb'))