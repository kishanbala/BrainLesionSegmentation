import os
import pickle

def prepare_train_data():
    '''
    Store the path of Input and corresponding output images in definite order
    :return: Lists containing input & output image directory locations
    '''
    HOME_DIR = 'P:\\WMH\\Train'
    files_list = os.listdir(HOME_DIR)

    patient_num = 60

    X = []
    y = []

    for i in range(0,patient_num):
        file = files_list.__getitem__(i)
        X_path = HOME_DIR + '\\' + file + '\\FLAIR'
        flair_imgs = os.listdir(X_path)

        Y_path = HOME_DIR + '\\' + file + '\\MASK'
        mask_imgs = os.listdir(Y_path)

        for img in flair_imgs:
            path = X_path + '\\' + img
            X.append(path)

        for img in mask_imgs:
            path = Y_path + '\\' + img
            y.append(path)

    pickle_out = open("X.pickle", 'wb')
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("Y.pickle", 'wb')
    pickle.dump(y, pickle_out)
    pickle_out.close() 