import os
import pandas as pd


os.chdir('/Volumes/Drive/Dataset-CMB/ADNI-CMBs/Output')
DIR = '/Volumes/Drive/Dataset-CMB/ADNI-CMBs/Output'

df = pd.read_csv('/Volumes/Drive/Dataset-CMB/ADNI-CMBs/meta/MAYOADIRL_MRI_MCH_08_15_19.csv', delimiter=",")
df = df[df['TYPE']=='MCH']
df_subject = pd.DataFrame(df.iloc[:, 1].copy())
df_scan_date = pd.DataFrame(df.iloc[:, 4].copy())
df_coordinates = pd.DataFrame(df.iloc[:, 17].copy())

new_df = df_subject.join(df_scan_date, how ='outer')
new_df = new_df.join(df_coordinates, how = 'outer')


output_list = []

for folder in os.listdir('.'):
    target = os.path.join(DIR, folder)
    os.chdir(target)

    for file in os.listdir('.'):
        file = file[:-4]
        subject = file[:-9]
        subject = int(subject)
        scan_date = file[-8:]
        scan_date = int(scan_date)

        for i in new_df[new_df.columns[0]]:
            if subject == i:
                new_df_temp = new_df[new_df['RID']==i]
                for j in new_df[new_df_temp.columns[1]]:
                    if scan_date == j:
                        new_df_temp = new_df_temp[new_df_temp['SCANDATE'] == j]
                        for k in new_df_temp[new_df_temp.columns[2]]:
                            #my_output = (str(i) + ',' + str(j) + ',' + str(k))
                            my_output = (str(k))
                            #output_list.append(my_output.split(","))
                            output_list.append(my_output)
                        break
                break

list_df = pd.DataFrame(output_list)
list_df.to_csv('/Volumes/Drive/Dataset-CMB/ADNI-CMBs/meta/final_output.csv', sep=',', index=False)











