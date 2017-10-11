def copy_data_excel():
    import xlwt
    import xlrd
    import math
    import csv
    import random
    import standardization
    import trial7

    standardization.standraziation()

    input_var = input("Percentage to copy from raw data as validation data (e.g., 0.8 for 80 percent): ")
    int_conv=int(float(input_var)*100)
    

    workbook = xlrd.open_workbook('standardized_data.xlsx')
    #workbook = xlrd.open_workbook('myfile.xls')
    sheet1 = workbook.sheet_by_name('standardized_data')
    total_rows=sheet1.nrows
    gg=random.sample(range(1,total_rows),math.ceil(total_rows*float(input_var)))
    print(len(gg))

    #data = [sheet1.cell_value(0, col) for col in range(sheet1.ncols)]
    #data1 = [sheet1.cell_value(row, 0) for row in range(math.ceil(sheet1.nrows*float(input_var)))]

    workbook = xlwt.Workbook()
    sheet2 = workbook.add_sheet('test_data')
    for index1 in range(0,math.ceil(total_rows*float(input_var))):
        for index in range(0,sheet1.ncols):
            y=gg[index1]
            value=sheet1.cell_value(y,index)
            sheet2.write(index1,index,value)
    workbook.save('test_data.xls')
    hh = []
    diff=total_rows-math.ceil(total_rows*float(input_var))

    for i in range(diff):
           hh.append(0)
    j=0   
    for i in range(0,total_rows):
        if i not in gg:
            
        #else:
            hh[j]=i
            j=j+1
    print(len(hh))
    print(len(gg))
    print(total_rows)

    workbook1 = xlwt.Workbook()
    sheet3 = workbook1.add_sheet('train_data')
    for index2 in range(0,len(hh)):
        for index in range(0,sheet1.ncols):
            yy=hh[index2]
            value=sheet1.cell_value(yy,index)
            sheet3.write(index2,index,value)

            
    #for index1 in range(0,math.ceil(sheet1.nrows*float(input_var))):
     #   for index in range(0,sheet1.ncols):
      #      value=sheet1.cell_value(index1,index)
       #     sheet2.write(index1,index,value)


    
    workbook1.save('train_data.xls')
  

copy_data_excel()
import trttt
trttt.xls_to_xsv()
import trial7
trial7.delete_first_col_csv()





