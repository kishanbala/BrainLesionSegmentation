import nibabel as nib
import numpy as np
import scipy.ndimage
import os
from PIL import Image , ImageOps

def normalize(input_image,image_name):
    # rows_standard = 200
    # cols_standard = 200

    image = nib.load(input_image)
    header = image.header
    affine = image.affine

    FLAIR_image = nib.load(input_image).get_fdata()
    FLAIR_image = np.array(FLAIR_image)


    FLAIR_image -= np.mean(FLAIR_image)
    FLAIR_image /= np.std(FLAIR_image)

    FLAIR_image = nib.Nifti1Image(FLAIR_image, affine=affine, header=header)
    nib.save(FLAIR_image, image_name+'.gz')
    return FLAIR_image



HOME_DIR = '/Volumes/Seagate Backup Plus Drive/Dataset-CMB/ADNI-CMBs/output/ADNI 4'

#normalize(HOME_DIR,'normalized')

for subjects in os.listdir(HOME_DIR):
    if subjects.endswith('.nii'):
        normalize(os.path.join(HOME_DIR, subjects), subjects)