import os
import re
import pandas as pd
import dicom2nifti as d2n

#d2n.convert_directory('/Drive/Dataset-CMB/ADNI-CMBs/data/ADNI/002_S_0685/Axial_T2-Star/2011-07-08_07_59_15.0/S114041/', '/Drive/Dataset-CMB/ADNI-CMBs/data/ADNI/002_S_0685/Output/S114041.nii', compression=True, reorient=True)
#d2n.dicom_series_to_nifti('/Volumes/Drive/Dataset-CMB/ADNI-CMBs/data/ADNI/002_S_0685/Axial_T2-Star/2011-07-08_07_59_15.0/"*s*"/','/Volumes/Drive/Dataset-CMB/ADNI-CMBs/data/ADNI/002_S_0685/Output/S114041.nii',reorient_nifti=True)
#d2n.dicom_series_to_nifti('/Users/lokesh/Desktop/CSVD/test','/Users/lokesh/Desktop/CSVD/Output/S114041.nii',reorient_nifti=True)

os.chdir('/Volumes/Drive/Dataset-CMB/ADNI-CMBs/data/')
DIR = '/Volumes/Drive/Dataset-CMB/ADNI-CMBs/data/'

df = pd.read_csv('/Volumes/Drive/Dataset-CMB/ADNI-CMBs/meta/MAYOADIRL_MRI_MCH_08_15_19.csv', delimiter="\t")
df_subject = df.iloc[:, 1].copy()
df_scan_date = df.iloc[:, 4].copy()
df_coordinates = df.iloc[:, 17].copy()


for main_folder in os.listdir('.'):
    output = '/Volumes/Drive/Dataset-CMB/ADNI-CMBs/Output'
    if ((main_folder.endswith(".zip") or (main_folder.endswith(".csv")) or main_folder == 'ADNI' or main_folder == 'ADNI 2' or main_folder == 'ADNI 3' or main_folder == 'ADNI 4' or main_folder == 'ADNI 10' or main_folder == 'ADNI 5' or main_folder == 'ADNI 6' or main_folder == 'ADNI 7') == False):
        os.chdir('/Volumes/Drive/Dataset-CMB/ADNI-CMBs/Output')
        os.mkdir(main_folder)
        output_temp = os.path.join(output, main_folder)
        main_target = os.path.join(DIR, main_folder)
        os.chdir(main_target)
        #print([name for name in os.listdir('.') if os.path.isdir(name)])

        for folder in os.listdir('.'):
        #filename = folder + '.nii'
            target = os.path.join(main_target, folder)
            os.chdir(target)

            for subfolder in os.listdir('.'):
                if (subfolder.endswith(".nii") == False):
                    newtarget = os.path.join(target, subfolder)
                    os.chdir(newtarget)

                    for nextsubfolder in os.listdir('.'):
                        filename_1 = folder[-4:]
                        filename = nextsubfolder
                        filename = filename[:-11]
                        filename = re.sub('[-]', '', filename)
                        filename = filename_1 + '_' + filename + '.nii'

                        output = os.path.join(output_temp, filename)
                        final_target = os.path.join(newtarget, nextsubfolder)
                        d2n.dicom_series_to_nifti(final_target, output, reorient_nifti=True)