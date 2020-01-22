import pickle
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

class GenericImageDataset():
    '''
    Incorporate different transformations as specified on the given images
    '''

    def __init__(self, X, y,  transformations):
        self.X = X
        self.y = y
        self.transformations = transformations

    def __getitem__(self, index):
        max_val = 1.0
        min_val = 0.0

        image = self.X[index]
        mask = self.y[index]

        image = Image.open(image)
        mask = Image.open(mask)

        image = np.array(image, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)

        if self.transformations is not None:
            image_tensor = self.transformations(image)
            mask_tensor = self.transformations(mask)

        return image_tensor, mask_tensor

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

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((200, 200),interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

    dset_train = GenericImageDataset(X_path,
                                     y_path,
                                     trans)

    train_data = []
    for index in range(0, len(dset_train.X)):
        image, label = dset_train.__getitem__(index)
        train_data.append([image, label])

    pickle_out = open("tensor.pickle", 'wb')
    pickle.dump(train_data, pickle_out)
    pickle_out.close()