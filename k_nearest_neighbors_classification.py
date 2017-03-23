import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
#reading and preparing data set
data_set = pd.read_table('k_neighbor_breast-cancer-wisconsin.data.txt',header=None,sep=',')
column = ['id','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','class']

data_set.columns = column
clms = data_set.columns.str.replace(' ','_')
data_set.columns = clms
#
data_set.replace('?',-99999,inplace=True)
# print data_set.head()

data_set.drop(['id'],1,inplace=True)

# define x and y
x= np.array(data_set.drop(['class'],1))
y=np.array(data_set['class'])

X_train, X_test , Y_train , Y_test = cross_validation.train_test_split(x,y)

classfy = neighbors.KNeighborsClassifier()
classfy.fit(X_train,Y_train)

confedincy = classfy.score(X_test,Y_test)
print confedincy