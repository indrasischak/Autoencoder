def xls_to_xsv():
    import pandas as pd
    data_xls = pd.read_excel('test_data.xls', 'test_data', index_col=None)
    data_xls.to_csv('test_data.csv', encoding='utf-8')

    data_xls1 = pd.read_excel('train_data.xls', 'train_data', index_col=None)
    data_xls1.to_csv('train_data.csv', encoding='utf-8')

    data_xls2 = pd.read_excel('predict_data.xls', 'predict_data', index_col=None)
    data_xls2.to_csv('predict_data.csv', encoding='utf-8')

