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
    traindata = np.zeros(shape=[4300,2])
    testdata_n = np.zeros(shape=[293,2])
    testdata_p = np.zeros(shape=[sheet1.nrows,4])
    for index1 in range(0,4299):
        for index in range(0,1):
            traindata[index1,index]=sheet1.cell_value(index1,index)

        #traindata[index1,6]=sheet1.cell_value(index1,6)
        #traindata[index1,7]=sheet1.cell_value(index1,7)
        #traindata[index1,8]=sheet1.cell_value(index1,8)
        #traindata[index1,9]=sheet1.cell_value(index1,9)
        #traindata[index1,4]=sheet1.cell_value(index1,4)

        #testdata_n[index1,0]=sheet1.cell_value(index1,6)
        #testdata_n[index1,1]=sheet1.cell_value(index1,7)
        #testdata_n[index1,2]=sheet1.cell_value(index1,8)
        #testdata_n[index1,3]=sheet1.cell_value(index1,9)
        #testdata_n[index1,4]=sheet1.cell_value(index1,10)
        #testdata_n[index1,5]=sheet1.cell_value(index1,11)
        #testdata_n[index1,6]=sheet1.cell_value(index1,14)
        #testdata_n[index1,7]=sheet1.cell_value(index1,15)
        #testdata_n[index1,8]=sheet1.cell_value(index1,18)
        #testdata_n[index1,9]=sheet1.cell_value(index1,19)

        #testdata_p[index1,0]=sheet1.cell_value(index1,10)
        #testdata_p[index1,1]=sheet1.cell_value(index1,11)
        #testdata_p[index1,2]=sheet1.cell_value(index1,12)
        #testdata_p[index1,3]=sheet1.cell_value(index1,13)
        #testdata_p[index1,4]=sheet1.cell_value(index1,14)

#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
    #traindata_new = traindata.reshape(len(traindata), 4)
    #testdata_new = testdata_n.reshape(len(testdata_n), 4)
    return traindata
