def ggg():
    import xlwt
    import xlrd
    import math
    import csv
    import random
    import numpy as np
    


    workbook = xlrd.open_workbook('tryy2.xlsx')
    #workbook = xlrd.open_workbook('myfile.xls')
    sheet1 = workbook.sheet_by_name('tryy2')
    traindata = np.zeros(shape=[sheet1.nrows,4])
    testdata_n = np.zeros(shape=[sheet1.nrows,4])
    testdata_p = np.zeros(shape=[sheet1.nrows,4])
    for index1 in range(0,sheet1.nrows):
        traindata[index1,0]=sheet1.cell_value(index1,0)
        traindata[index1,1]=sheet1.cell_value(index1,1)
        traindata[index1,2]=sheet1.cell_value(index1,2)
        traindata[index1,3]=sheet1.cell_value(index1,3)
        #traindata[index1,4]=sheet1.cell_value(index1,4)

        testdata_n[index1,0]=sheet1.cell_value(index1,4)
        testdata_n[index1,1]=sheet1.cell_value(index1,5)
        testdata_n[index1,2]=sheet1.cell_value(index1,6)
        testdata_n[index1,3]=sheet1.cell_value(index1,7)
        #testdata_n[index1,4]=sheet1.cell_value(index1,9)

        #testdata_p[index1,0]=sheet1.cell_value(index1,10)
        #testdata_p[index1,1]=sheet1.cell_value(index1,11)
        #testdata_p[index1,2]=sheet1.cell_value(index1,12)
        #testdata_p[index1,3]=sheet1.cell_value(index1,13)
        #testdata_p[index1,4]=sheet1.cell_value(index1,14)

#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
    traindata_new = traindata.reshape(len(traindata), 4)
    testdata_new = testdata_n.reshape(len(testdata_n), 4)
    return testdata_new
