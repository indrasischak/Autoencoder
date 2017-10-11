import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import NuSVC
import data_gather2
import xlwt
import random
mnist1=data_gather2.ggg()
X = mnist1[:, :5] 
            
y = mnist1[:,6]
#X_new = [[[0]*1999]*5]
#y_new=np.zeros(1999)
dum=random.sample(range(0,3997),2000)
for i in range(0,1999):
    for j in range(0,5):
        X_new[i,j]=X[dum[i],j]
        y_new[i]=y[dum[i]]
    


clf = NuSVC()
clf.fit(X_new, y_new) 
NuSVC(cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=1, gamma='auto', kernel='rbf',
      max_iter=-1, nu=0.5, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)
y1=clf.predict(X)
workbook = xlwt.Workbook()
sheet2 = workbook.add_sheet('svm_result')
for index1 in range(0,3997):
        value=y1[index1]
        sheet2.write(index1,1,np.float(value))
workbook.save('svm_result.xls')
plt.plot(y1)
plt.plot(y)
plt.show()
