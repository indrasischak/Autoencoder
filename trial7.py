def delete_first_col_csv():
    import csv

    with open('test_data.csv',"r") as fin:
        with open('test_new.csv',"w",newline='') as fout:
            writer=csv.writer(fout)
            for col in csv.reader(fin):
                writer.writerow(col[1:])

    with open('train_data.csv',"r") as fin:
        with open('train_new.csv',"w",newline='') as fout:
            writer=csv.writer(fout)
            for col in csv.reader(fin):
                writer.writerow(col[1:])

    with open('predict_data.csv',"r") as fin:
        with open('predict_new.csv',"w",newline='') as fout:
            writer=csv.writer(fout)
            for col in csv.reader(fin):
                writer.writerow(col[1:])

