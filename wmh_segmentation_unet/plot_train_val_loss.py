import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('P:\\WMH\\CHECKPOINTS\\WITHOUT_AUG\\plot_loss_80_20.txt', delimiter=',')
#print(df.head())

newdf = pd.DataFrame()

for index,data in df.iterrows():
    #print(data.shape[0])
    for i in range(0,data.shape[0]):
        data[i] = str(data[i]).replace('[','')
        data[i] = str(data[i]).replace(']', '')
        newdf.loc[index,i] = data [i]

newdf.to_csv('loss_result_80_20.csv',sep=',')






