import os
from glob import glob
import nibabel as nib
import numpy as np
from PIL import Image

def split_filename(filepath):
    '''
    :param filepath: NifTI file directory
    :return: Splits the filename and extension, this filename is used as Subject ID to store FLAIR & GT images
    '''
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext

def convert_to_slice(img_dir, out_dir, flair, mask, min_range = 0, max_range = 1, axis = 2):
    '''
    :param img_dir: FLAIR directory containing input 2D slices of actual image
    :param out_dir: MASK directory containing output 2D slices of segmented ground truth
    :param flair: Indicating if the current input passed is FLAIR/input image
    :param mask: Indicating if the current input passed is ground_truth/output image
    :param min_range: Starting index range to be sliced
    :param max_range: Ending index range to be sliced
    :param axis: Here Z axis is chosen as default since input is 3D images
    :return: Save 2D slices along the given axis and slicing range in TIFF format
    '''
    if flair:
        fns = glob(os.path.join(img_dir, 'FLAIR.nii*'))
    if mask:
        fns = glob(os.path.join(img_dir, 'wmh.nii*'))

    for fn in fns:
        _, base, ext = split_filename(fn)
        img = nib.load(fn).get_data().astype(np.float32).squeeze()
        if img.ndim != 3:
            print(f'Only 3D data supported. File {base}{ext} has dimension {img.ndim}. Skipping.')
            continue
        start = int(min_range * img.shape[axis])
        end = int(max_range * img.shape[axis])
        for i in range(start, end):
            I = Image.fromarray(img[:,:,i], mode='F')
            I.save(os.path.join(out_dir, f'{base}_{i:04}.tif'))

        if flair:
            print('FLAIR slices created ', out_dir)
        if mask:
            print('MASK slices created ', out_dir)

def folder_to_convert():
    '''
    Input : 60 subjects ( 20 subjects for each hospital/scanner)
    Operation:  Take 2D axial slices for each subject
                Ignoring 20% of slices in the first and last portions since they do not contain WMHs
                FLAIR and WMH folders alone considered from given dataset - Input & Output
    :return: 60 folders containing FLAIR as input & MASK as output with matching number of 2D axial slices
             This is treated as (X,y) for data preparation
    '''
    patient_num = 60

    for i in range(0, patient_num):
        if i < 20:
            IN_DIR = 'P:\\WMH\\Utrecht\\Utrecht'
            OUT_DIR = 'P:\\WMH\\'
            dirs = os.listdir(IN_DIR)
            a = dirs.__getitem__(i)
            path = IN_DIR + '\\' + a + '\\pre'
            img_dir = os.path.join(path)

            mask_path = IN_DIR + '\\' + a
            mask_dir = os.path.join(mask_path)

            out_dir = OUT_DIR + a
            os.mkdir(out_dir)
            print(out_dir + ' is created.')

            FLAIR_out_dir = out_dir+'\\FLAIR'
            os.mkdir(FLAIR_out_dir)

            MASK_out_dir = out_dir+'\\MASK'
            os.mkdir(MASK_out_dir)

            convert_to_slice(img_dir, FLAIR_out_dir, flair=True, mask=False, min_range=0.1, max_range=0.9)
            convert_to_slice(mask_dir, MASK_out_dir, flair= False, mask= True, min_range=0.1, max_range=0.9)

        elif (i>=20) and (i<40):
            IN_DIR = 'P:\\WMH\\Singapore\\Singapore'
            OUT_DIR = 'P:\\WMH\\'

            dirs = os.listdir(IN_DIR)
            a = dirs.__getitem__(i%20)
            path = IN_DIR + '\\' + a + '\\pre'
            img_dir = os.path.join(path)

            mask_path = IN_DIR + '\\' + a
            mask_dir = os.path.join(mask_path)

            out_dir = OUT_DIR + a
            os.mkdir(out_dir)
            print(out_dir + ' is created.')

            FLAIR_out_dir = out_dir + '\\FLAIR'
            os.mkdir(FLAIR_out_dir)

            MASK_out_dir = out_dir + '\\MASK'
            os.mkdir(MASK_out_dir)

            convert_to_slice(img_dir, FLAIR_out_dir, flair=True, mask=False, min_range=0.1, max_range=0.9)
            convert_to_slice(mask_dir, MASK_out_dir, flair=False, mask=True, min_range=0.1, max_range=0.9)

        else:
            IN_DIR = 'P:\\WMH\\Amsterdam_GE3T\\GE3T'
            OUT_DIR = 'P:\\WMH\\'

            dirs = os.listdir(IN_DIR)
            a = dirs.__getitem__(i%40)
            path = IN_DIR + '\\' + a + '\\pre'
            img_dir = os.path.join(path)

            mask_path = IN_DIR + '\\' + a
            mask_dir = os.path.join(mask_path)

            out_dir = OUT_DIR + a
            os.mkdir(out_dir)
            print(out_dir + ' is created.')

            FLAIR_out_dir = out_dir + '\\FLAIR'
            os.mkdir(FLAIR_out_dir)

            MASK_out_dir = out_dir + '\\MASK'
            os.mkdir(MASK_out_dir)

            convert_to_slice(img_dir, FLAIR_out_dir, flair=True, mask=False, min_range=0.1, max_range=0.9)
            convert_to_slice(mask_dir, MASK_out_dir, flair=False, mask=True, min_range=0.1, max_range=0.9)