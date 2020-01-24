import os
import pickle

def store_path():
    X = []
    y = []

    IMG = 'F:\\CMB\\nii_patches'
    image_files = os.listdir(IMG)

    for file in image_files:
        dir = os.path.join(IMG,file)
        for paths in os.listdir(dir):
            X.append(os.path.join(dir,paths))

    GT = 'F:\\CMB\\gt_patches'
    image_files = os.listdir(GT)

    for file in image_files:
        dir = os.path.join(IMG, file)
        for paths in os.listdir(dir):
            y.append(os.path.join(dir, paths))

    pickle_out = open("input_paths.pickle", 'wb')
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("gt_paths.pickle", 'wb')
    pickle.dump(y, pickle_out)
    pickle_out.close()

store_path()