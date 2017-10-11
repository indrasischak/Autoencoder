import numpy as np
import xlwt
import xlrd
import math
import csv
import random
    

workbook = xlrd.open_workbook('Dymoladata_summer.xlsx')
    #workbook = xlrd.open_workbook('myfile.xls')
sheet1 = workbook.sheet_by_name('Dymoladata_summer')
y=np.zeros(shape=(sheet1.nrows,sheet1.ncols)) 
for index in range(0,sheet1.nrows):
    for index1 in range(0,sheet1.ncols):
        y[index,index1]=sheet1.cell_value(index,index1)

