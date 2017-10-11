from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import NuSVC
import data_gather
import data_gather4
import xlwt
import random
mnist1=data_gather.ggg()
mnist2=data_gather4.ggg()
XX=np.concatenate((mnist1,mnist2))
np.random.shuffle(XX)
#input_var = input("Number of data fold:")
size=len(XX);
y=np.zeros(shape=[size,1])
for index in range(0,size):
    if (4321-index)>=0:
        y[index,0]=0
    else:
        y[index,0]=1
kf = KFold(n_splits=20)
kf.get_n_splits(XX)
print(kf)

KFold(n_splits=20, random_state=None, shuffle=False)
ii=0
dum=np.zeros(len(XX))
for train_index, test_index in kf.split(XX):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = XX[train_index], XX[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf=NuSVC()
        clf.fit(X_train,y_train)
        NuSVC(cache_size=200, class_weight=None, coef0=0.0,
          decision_function_shape=None, degree=1, gamma='auto', kernel='rbf',
          max_iter=-1, nu=0.5, probability=False, random_state=None,
          shrinking=True, tol=0.001, verbose=False)
        y_predict=clf.predict(X_test)
        for i in range(0,len(y_predict)-1):
            if y_predict[i]==y_test[i]:
                dum[ii]=1
                ii=ii+1
            else:
                dum[ii]=0
                ii=ii+1
print("Accuracy is")
print((np.sum(dum)/len(XX))*100,"%")


