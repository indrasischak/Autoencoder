import csv

path = 'C:/Users/chak282/AppData/Local/Programs/Python/Python35/'

reader = csv.reader(open(path +"predict_new.csv", "r"), delimiter=',')
 
data = []
 
for row in reader:
    data.append(row)
 
#now you can use indices to address a cell such as:
print(data[0][1])
