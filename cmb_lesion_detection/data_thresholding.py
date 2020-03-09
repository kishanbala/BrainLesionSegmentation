import nibabel as nib
import numpy as np
import os
import random


# input_dir = '/Volumes/Seagate Backup Plus Drive/Dataset-CMB/ADNI-CMBs/output/ADNI 4/test_in'
# gt_dir = '/Volumes/Seagate Backup Plus Drive/Dataset-CMB/ADNI-CMBs/output/ADNI 4/test_gt'

def balanced_dataset(input_dir, gt_dir):

    positive_list = []
    negative_list = []

    count_pos = 0
    count_neg = 0

    input = os.listdir(input_dir)
    gt = os.listdir(gt_dir)
    input.sort()
    gt.sort()

    for x, y in zip(input, gt):
        input_sub_dir = os.path.join(input_dir, x)
        gt_sub_dir = os.path.join(gt_dir, y)

        input_sub_dir_patches = os.listdir(input_sub_dir)
        gt_sub_dir_patches = os.listdir(gt_sub_dir)

        input_sub_dir_patches.sort()
        gt_sub_dir_patches.sort()

        for in_patch, gt_patch in zip(input_sub_dir_patches, gt_sub_dir_patches):
            input_image = os.path.join(input_sub_dir, in_patch)
            gt_image = os.path.join(gt_sub_dir, gt_patch)

            FLAIR_image_in = nib.load(input_image).get_fdata()
            FLAIR_image_in = np.array(FLAIR_image_in)

            FLAIR_image_gt = nib.load(gt_image).get_fdata()
            FLAIR_image_gt = np.array(FLAIR_image_gt)

            if FLAIR_image_gt.max() == 1.0:
                positive_list.append((FLAIR_image_in, FLAIR_image_gt))
                count_pos +=1
                print('gt : ' + gt_patch)
            else:
                negative_list.append((FLAIR_image_in, FLAIR_image_gt))
                count_neg +=1


    positive_count  = len(positive_list)
    negative_list_1 = random.sample(negative_list, positive_count)

    balanced_list = positive_list + negative_list_1
    print(len(positive_list))
    print(len(negative_list))

    return balanced_list