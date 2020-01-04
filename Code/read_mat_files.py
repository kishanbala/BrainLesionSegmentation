import scipy.io as sci
import os

DIR = 'P:\\CSVD_Detection\\Cerebral_Microbleeds\\State_of_the_art_technique\\Dataset\\cmb-3dcnn-data\\ground_truth'

for file in os.listdir(DIR):

    input = os.path.join(DIR, file)
    x = sci.loadmat(input)
    cen_data = x['cen']
    gt_num_data = x['gt_num']
    print(file)
    print(cen_data)
    print(gt_num_data)