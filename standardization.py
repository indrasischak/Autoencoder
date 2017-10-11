def standraziation():
    import numpy as np
    import xlwt
    import xlrd
    import math
    import csv
    import random
    

    workbook = xlrd.open_workbook('Dymoladata_summer.xlsx')
    #workbook = xlrd.open_workbook('myfile.xls')
    sheet1 = workbook.sheet_by_name('Dymoladata_summer')
    y = []
    mean_val=[]
    std_val=[]
    for i in range(sheet1.nrows):
           y.append(0)
    for i in range(sheet1.ncols):
        mean_val.append(0)
        std_val.append(0)
    for index in range(0,sheet1.ncols):
        for index1 in range(0,sheet1.nrows):
            y[index1]=sheet1.cell_value(index1,index)
            mean_val[index]=np.mean(y)
            std_val[index]=np.std(y)

    print(mean_val)
    print(std_val)

    workbook1 = xlwt.Workbook()
    sheet2 = workbook1.add_sheet('standardized_data')
    for index1 in range(0,sheet1.nrows):
        for index in range(0,sheet1.ncols):
            value=(sheet1.cell_value(index1,index)-mean_val[index])/std_val[index]
            sheet2.write(index1,index,value)
    workbook1.save('standardized_data.xlsx')



    workbook2 = xlrd.open_workbook('faulty_data.xlsx')
    #workbook = xlrd.open_workbook('myfile.xls')
    sheet2 = workbook2.sheet_by_name('faulty_data')


    workbook3 = xlwt.Workbook()
    sheet3 = workbook3.add_sheet('predict_data')
    for index1 in range(0,sheet2.nrows):
        for index in range(0,sheet2.ncols):
            value=(sheet2.cell_value(index1,index)-mean_val[index])/std_val[index]
            sheet3.write(index1,index,value)
    workbook3.save('predict_data.xls')



