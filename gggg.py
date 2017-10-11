import xlwt
import xlrd
import math
import csv
import random
import numpy as np
    

workbook = xlrd.open_workbook('tryy.xlsx')
    #workbook = xlrd.open_workbook('myfile.xls')
sheet1 = workbook.sheet_by_name('tryy')
y = np.zeros(shape=[sheet1.nrows,1])
for index1 in range(0,sheet1.nrows):
    y[index1,0]=sheet1.cell_value(index1,0)
