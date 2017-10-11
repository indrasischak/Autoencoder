from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import xlrd
import random
import math

def rnn_data(data, time_steps, labels=False):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]]
        -> labels == True [3, 4, 5]
    """
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:
            try:
                rnn_df.append(data.iloc[i + time_steps].as_matrix())
            except AttributeError:
                rnn_df.append(data.iloc[i + time_steps])
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

    return np.array(rnn_df, dtype=np.float32)

workbook = xlrd.open_workbook('RNN_data.xlsx')
sheet1 = workbook.sheet_by_name('RNN_data')
data=np.zeros(shape=[sheet1.nrows,2])

for index1 in range(0,sheet1.nrows):
    for index in range(0,1):
        data[index1,index]=sheet1.cell_value(index1,index)

input_var = input("Percentage to copy from raw data as validation data (e.g., 0.8 for 80 percent): ")
time_steps=3

int_conv=int(float(input_var)*100)
    
total_rows=sheet1.nrows
gg=random.sample(range(1,total_rows),math.ceil(total_rows*float(input_var)))
test_data=np.zeros(shape=[math.ceil(total_rows*float(input_var)),2])
for index1 in range(0,math.ceil(total_rows*float(input_var))):
    for index in range(0,1):
        y=gg[index1]
        test_data[index1,index]=data[y,index]
    
hh=[]            
diff=total_rows-math.ceil(total_rows*float(input_var))
for i in range(diff):
    hh.append(0)
j=0   
for i in range(0,total_rows):
    if i not in gg:
            
        #else:
            hh[j]=i
            j=j+1
train_data=np.zeros(shape=[len(hh),2])    
for index1 in range(0,len(hh)):
    for index in range(0,1):
        yy=hh[index1]
        train_data[index1,index]=data[yy,index]
        
train_data_mod=pd.DataFrame(train_data,columns=list('XY'))
test_data_mod=pd.DataFrame(test_data,columns=list('XY'))

train_data_X=rnn_data(train_data_mod['X'],time_steps,labels=False)
train_data_Y=rnn_data(train_data_mod['Y'],time_steps,labels=False)

test_data_X=rnn_data(test_data_mod['X'],time_steps,labels=False)
test_data_Y=rnn_data(test_data_mod['Y'],time_steps,labels=False)

predict_data_X=test_data_X
predict_data_Y=test_data_Y

def generate_data(train_data_X,train_data_Y,test_data_X,test_data_Y,predict_data_X,predict_data_Y):
        return dict(train=train_data_X, val=test_data_X, test=predict_data_X), dict(train=train_data_Y, val=test_data_Y, test=predict_data_Y)

X,y=generate_data(train_data_X,train_data_Y,test_data_X,test_data_Y,predict_data_X,predict_data_Y)


