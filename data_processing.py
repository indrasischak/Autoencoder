import pandas as pd
df = pd.read_csv("C:/Users/chak282/AppData/Local/Programs/Python/Python35/Dymoladata_summer.csv") 

df=df.drop(df.index[[0]])
#df1= df.convert_objects(convert_numeric=True)
