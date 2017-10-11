import tensorflow as tf
import numpy as np
    #import data_gather_sin1_auto
from random import randint
import xlwt
import xlrd
import math
import csv
import random
import numpy as np
workbook = xlrd.open_workbook('tryy2.xlsx')
    #workbook = xlrd.open_workbook('myfile.xls')
sheet1 = workbook.sheet_by_name('tryy2')
total_train_row=10000
training_epochs=100
traindata = np.zeros(shape=[10000,5])
testdata_n = np.zeros(shape=[10000,5])
testdata_p = np.zeros(shape=[sheet1.nrows,4])
for index1 in range(0,10000):
    traindata[index1,0]=sheet1.cell_value(index1,0)
    traindata[index1,1]=sheet1.cell_value(index1,1)
    traindata[index1,2]=sheet1.cell_value(index1,2)
    traindata[index1,3]=sheet1.cell_value(index1,3)
    traindata[index1,4]=sheet1.cell_value(index1,4)
mnist=traindata
mnist1=traindata
