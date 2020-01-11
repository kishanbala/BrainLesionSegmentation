import pickle
from torchvision import transforms
from PIL import Image
import numpy as np

class GenericImageDataset():
    '''
    Incorporate different transformations as specified on the given images
    '''

    def __init__(self, X, y, transformations):
        self.X = X
        self.y = y
        self.transformations = transformations

    def __getitem__(self, index):

        image = self.X[index]
        mask = self.y[index]

        image = Image.open(image)
        mask = Image.open(mask)

        image = np.array(image, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)

        if self.transformations is not None:
            image_tensor = self.transformations(image)
            mask_tensor = self.transformations(mask)

        return  image_tensor, mask_tensor

def data_to_dim_shape():
    '''
    Input: Image and ground truth locations as lists
    Operation:  Open the images and resize to 200 x 200, this is considered as original list of images
                Rotate the images by 15 degrees and append to the original list of images
                Scale the images by 10% and append to the original list of images
                Shear the images by 18 degrees and append to the original list of images
                Convert all these images to Tensor
    :return:
    '''
    pickle_in = open('X.pickle', 'rb')
    X_path = pickle.load(pickle_in)

    pickle_in = open('y.pickle', 'rb')
    y_path = pickle.load(pickle_in)


    ## RESIZE
    trans1 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((200,200)),
        transforms.ToTensor()
    ])

    ## RESIZE AND ROTATE
    trans2 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((200,200)),
        transforms.RandomAffine(degrees=[-15, 15]),
        transforms.ToTensor()
    ])

    ## RESIZE AND SCALE
    trans3 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((200,200)),
        transforms.RandomAffine(degrees=0, scale=[0.9, 1.1]),
        transforms.ToTensor()
    ])

    ## RESIZE AND SHEAR
    trans4 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((200,200)),
        transforms.RandomAffine(degrees=0, shear=[-18, 18]),
        transforms.ToTensor()
    ])

    ## APPENDING RESIZED IMAGES
    dset_train = GenericImageDataset(X_path,
                                     y_path,
                                     trans1)

    train_data = []
    for index in range(0, len(dset_train.X)):
        image, label = dset_train.__getitem__(index)
        #break
        train_data.append([image, label])

    ## APPENDING ROTATED IMAGES
    dset_train = GenericImageDataset(X_path,
                                     y_path,
                                     trans2)

    for index in range(0, len(dset_train.X)):
        image, label = dset_train.__getitem__(index)
        #break
        train_data.append([image, label])

    ## APPENDING SCALED IMAGES
    dset_train = GenericImageDataset(X_path,
                                     y_path,
                                     trans3)


    for index in range(0, len(dset_train.X)):
        image, label = dset_train.__getitem__(index)
        #break
        train_data.append([image, label])

    ## APPENDING SHEARED IMAGES
    dset_train = GenericImageDataset(X_path,
                                     y_path,
                                     trans4)


    for index in range(0, len(dset_train.X)):
        image, label = dset_train.__getitem__(index)
        #break
        train_data.append([image, label])

    print(len(train_data))

    pickle_out = open("tensor.pickle", 'wb')
    pickle.dump(train_data, pickle_out)
    pickle_out.close()