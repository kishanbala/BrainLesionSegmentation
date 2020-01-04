import os
import nibabel as nib
import numpy as np
import scipy.io as sci

IMAGE_PATH = 'P:\\CSVD_Detection\\Cerebral_Microbleeds\\State_of_the_art_technique\\Dataset\\cmb-3dcnn-data\\nii'
image_files = os.listdir(IMAGE_PATH)

GT_PATH = 'P:\\CSVD_Detection\\Cerebral_Microbleeds\\State_of_the_art_technique\\Dataset\\cmb-3dcnn-data\\ground_truth\\'


def split_filename(filepath):
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.nii':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext

for image in image_files:
    np_image = nib.load(filename=os.path.join(IMAGE_PATH,image)).get_data().astype(np.float32).squeeze()
    build_gt = np.zeros(np_image.shape)
    #print(build_gt.shape)
    path, base, ext = split_filename(os.path.join(IMAGE_PATH,image))
    gt_file = base + '.mat'
    gt_file = GT_PATH + gt_file

    x = sci.loadmat(gt_file)
    cen_data = x['cen']
    gt_num_data = x['gt_num']
    gt_num_data = int(gt_num_data)

    for cmb_count in range(0,gt_num_data):
        cmb_pos = cen_data[cmb_count]
        print(cmb_pos)
        build_gt[183:183, 215:215, 103:103] = 1.0

    new_image = nib.Nifti1Image(build_gt, affine=np.eye(4))
    save_path = GT_PATH + base + ext
    nib.save(new_image,save_path)

        #build_gt[cmb_pos] = 1.0
    #print(gt_file)
    break
