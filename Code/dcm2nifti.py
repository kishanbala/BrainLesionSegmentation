# Author: Harish Kumar Harihara Subramanian
# execute it like python dcm2nifti.py 'D:\\test_xi27' 'D:\\test_xi27_nifti.nii'

import dicom2nifti as d2n
import argparse
import sys

def arg_parser():
    parser = argparse.ArgumentParser(description='split 3d image into multiple 2d images')

    parser.add_argument('input_dicom_dir', type=str,
                        help='path to dicom image directory')

    parser.add_argument('out_nifti_file', type=str,
                        help='path to output nifti file')

    return parser

def main():
    try:
        args = arg_parser().parse_args()
        d2n.dicom_series_to_nifti(args.input_dicom_dir,
                                  args.out_nifti_file,
                                  reorient_nifti=True)
        return 0
    except Exception as exc:
        print(exc.args)
        return 1

if __name__ == "__main__":
    sys.exit(main())