
import sys
import pylab as plt
import numpy as np
plt.ion()

def show(X, C, centroids, keep = False):
    import time
    time.sleep(0.5)
    plt.cla()
    plt.plot(X[C == 0, 0], X[C == 0, 1], '*b',
         X[C == 1, 0], X[C == 1, 1], '*r',
         X[C == 2, 0], X[C == 2, 1], '*g')
    plt.plot(centroids[:,0],centroids[:,1],'*m',markersize=20)
    plt.draw()
    if keep :
        plt.ioff()
        plt.show()

# generate 3 cluster data
# data = np.genfromtxt('data1.csv', delimiter=',')
import data_gather2
X=data_gather2.ggg()
np.random.shuffle(X)

from kMeans import kMeans
centroids, C = kMeans(X, K = 5, plot_progress = show)
show(X, C, centroids, True)
