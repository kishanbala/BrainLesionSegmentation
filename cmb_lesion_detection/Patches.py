

from glob import glob
import nibabel as nib
import os
import numpy as np
import itk
from matplotlib import pyplot as plt
import sys

#img_dir = 'F:/CMB/nii'
img_dir = '/Volumes/Seagate Backup Plus Drive/Dataset-CMB/ADNI-CMBs/output/ADNI 4/groundtruth'
out_dir = '/Volumes/Seagate Backup Plus Drive/Dataset-CMB/ADNI-CMBs/output/ADNI 4/gt_patches'
x_shape = 8
y_shape = 8
z_shape = 11


# def arg_parser():
#     parser = argparse.ArgumentParser(description='split 3d image into multiple 3d patches')
#     parser.add_argument('img_dir', type=str,
#                         help='path to nifti image directory')
#     parser.add_argument('out_dir', type=str,
#                         help='path to output corresponding nifti patches')
#     parser.add_argument('x_shape', type=int,
#                         help='desired patch size')
#     parser.add_argument('y_shape', type=int,
#                         help='desired patch size')
#     parser.add_argument('z_shape', type=int,
#                         help='desired patch size')
#     return parser


def split_filename(filepath):
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext


def cut_3d_image(index_x, index_y, index_z, size_x, size_y, size_z, x_stride, y_stride, z_stride, img, image,
                 output_filename, base_name):
    stop = False
    output_path_folder = os.path.join(output_filename, base_name)
    # os.mkdir(output_filename + '\\' + base_name)
    os.mkdir(output_path_folder)

    x_offset = img.shape[0] - x_shape
    y_offset = img.shape[1] - y_shape
    z_offset = img.shape[2] - z_shape

    while (stop == False):

        if (index_x <= x_offset and index_y <= y_offset and index_z <= z_offset):
            cropper = itk.ExtractImageFilter.New(Input=image)
            cropper.SetDirectionCollapseToIdentity()
            extraction_region = cropper.GetExtractionRegion()

            size = extraction_region.GetSize()
            size[0] = int(size_x)
            size[1] = int(size_y)
            size[2] = int(size_z)

            index = extraction_region.GetIndex()
            index[0] = int(index_x)
            index[1] = int(index_y)
            index[2] = int(index_z)

            extraction_region.SetSize(size)
            extraction_region.SetIndex(index)
            cropper.SetExtractionRegion(extraction_region)

            output = output_filename + '/' + base_name + '/' + str(base_name) + '_' + str(index_x) + '_' + str(
                index_y) + '_' + str(index_z) + '.nii'
            itk.imwrite(cropper, output)
            #itk.ImageFileWriter.New(Input=cropper, FileName=output).Update()

        # now x,y,z index are 0
        # will cut one layer after another (top to bottom)
        # keeping y = 0, for every increase of x by 16 pos, iterate z-axis to 100 in installments of 10
        # i.e when z becomes 100, increment x by 16 and put back z as 0
        # if x reaches 512, then set x = 0, z = 0 but increment y by 16 pos

        if index_x <= x_offset:
            if index_z < z_offset:
                index_z += z_stride

            else:
                index_x += x_stride
                index_z = 0

        else:

            if index_x > x_offset:
                if index_y > y_offset:
                    stop = True

                else:
                    index_x = 0

            if index_y < y_offset:
                index_y += 16
                index_x = 0
                index_z = 0
            else:
                stop = True


#def main():
fns = glob(os.path.join(img_dir, '*.nii*'))
for fn in fns:
    _, base, ext = split_filename(fn)
    img = nib.load(fn).get_data().astype(np.float32).squeeze()
    if img.ndim != 3:
        print(f'Only 3D data supported. File {base}{ext} has dimension {img.ndim}. Skipping.')
        continue

    input_filename = fn

    reader = itk.imread(input_filename, itk.F)
    reader.Update()
    image = reader

    if (img.shape[0] % int(x_shape)) != 0:
        print(f'File {base}{ext} does not ensure equal split of input image along x axis')
        continue

    if (img.shape[1] % int(y_shape)) != 0:
        print(f'File {base}{ext} does not ensure equal split of input image along y axis')
        continue

    if (img.shape[2] % int(z_shape)) != 0:
        print(f'File {base}{ext} does not ensure equal split of input image along z axis')
        continue

    x_stride = x_shape
    y_stride = y_shape
    z_stride = z_shape

    index_x = 0
    index_y = 0
    index_z = 0

    size_x = x_shape
    size_y = y_shape
    size_z = z_shape

    cut_3d_image(index_x, index_y, index_z, size_x, size_y, size_z, x_stride, y_stride,
                 z_stride, img, image, out_dir, base)