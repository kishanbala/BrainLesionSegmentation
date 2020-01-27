import nibabel as nib
import numpy as np
import os
from torchvision import transforms
import torch.utils.data as data
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pickle

IMG = 'P:\\CMB\\Public_Dataset\\Input'
GT = 'P:\\CMB\\Public_Dataset\\GT'

inputs = os.listdir(IMG)
outputs = os.listdir(GT)

## READ A NIFTI IMAGE


## GETTING IMAGE SHAPE
# print(image.shape)
# print(image.min())
# print(image.max())
# print(type(image))

#ROLL A SLIDING WINDOW OF SHAPE 61x61


def read_sw():

    imgs_from_SW = []
    for file in inputs:

        input_image = nib.load(os.path.join(IMG, file)).get_data()
        output_image = nib.load(os.path.join(GT, file)).get_data()

        x, y, z = input_image.shape

        index_x = 0
        index_y = 0
        index_z = 0

        size_x = 61
        size_y = 61
        size_z = 1

        stride_x = 61
        stride_y = 61

        while (True):
            if (index_x < x - stride_x):
                # +1 because stride is 1 for sliding window

                # print(str(index_x) + ':' +str(size_x) + ',' +
                #       str(index_y) + ':' + str(size_y) + ',' +
                #       str(index_z) + ':' + str(size_z))

                clip_portion_in = input_image[index_x:size_x, index_y:size_y, index_z:size_z]
                clip_portion_out = output_image[index_x:size_x, index_y:size_y, index_z:size_z]
                imgs_from_SW.append([clip_portion_in, clip_portion_out])
                index_x = index_x + 1
                size_x = size_x + 1

            else:

                if (index_y < y - stride_y and size_y < y - stride_y):
                    index_x = 0
                    size_x = 61
                    index_y = index_y + stride_y
                    size_y = size_y + stride_y
                else:
                    # increment in Z direction
                    if (size_z < z):
                        index_x = 0
                        index_y = 0

                        size_x = 61
                        size_y = 61

                        index_z = index_z + 1
                        size_z = size_z + 1

                    else:
                        break
    return imgs_from_SW


twoD_images = read_sw()

class GenericImageDataset():
    '''
    Incorporate different transformations as specified on the given images
    '''

    def __init__(self, data,  transformations):
        self.data = data
        self.transformations = transformations

    def __getitem__(self, index):
        image, gt = self.data[index]

        if self.transformations is not None:
            image_tensor = self.transformations(image)
            gt_tensor = self.transformations(gt)

        return image_tensor, gt_tensor

## CONVERT THE DATA INTO TENSOR FOR DATA LOADER

def data_transformation():
    transformation = transforms.ToTensor()

    dset_train = GenericImageDataset(twoD_images,
                                     transformation)
    tensor_dataset = []
    for index in range(0, len(dset_train.data)):
        image, label = dset_train.__getitem__(index)
        tensor_dataset.append([image, label])

    return tensor_dataset

dataset = data_transformation()

# print(len(dataset))
#
# img, gt = dataset[1000]
# print(img.shape)
#
# img_ = np.squeeze(img)
# plt.imshow(img_)
# plt.show()

## SPLIT THE DATASET INTO TRAIN, VAL & TEST DATASETS (0.6, 0.3, 0.1)

def train_test_split(dset):
    length = len(dset)

    n_train = int(length * 0.6)
    n_test = int(length * 0.1)
    idx = list(range(length))

    train_idx = idx[: n_train]
    val_idx = idx[n_train: (n_train + n_test)]
    test_idx = idx[(n_train + n_test):]

    train_set = data.Subset(dset, train_idx)
    val_set = data.Subset(dset, val_idx)
    test_set = data.Subset(dset, test_idx)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=100, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

    #print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, val_loader, test_loader

#train, val, test = train_test_split(dataset)

# print(len(train))
# print(len(val))
# print(len(test))

class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16 , kernel_size=7, stride=1, padding=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=5, stride=5, padding=0))
            #torch.nn.Dropout(p=1 - keep_prob))

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=3, padding=0))
            #torch.nn.Dropout(p=1 - keep_prob))

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            #torch.nn.Dropout(p=1 - keep_prob))


        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        # torch.nn.Dropout(p=1 - keep_prob))

        self.fc1 = torch.nn.Linear(in_features=128*3*3, out_features=64, bias=True)
            #torch.nn.Dropout(p=1 - keep_prob))
        # L5 Final FC 625 inputs -> 10 outputs
        self.fc2 = torch.nn.Linear(in_features=64, out_features=2, bias=False)


    def forward(self, x):
        out = self.layer1(x)
        print('After layer 1: ', out.shape)

        out = self.layer2(out)
        print('After layer 2: ', out.shape)

        out = self.layer3(out)
        print('After layer 3: ', out.shape)

        out = self.layer4(out)
        print('After layer 4: ', out.shape)

        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.fc1(out)
        out = self.fc2(out)
        return out


## instantiate CNN model
model = CNN().cuda()
model.train()

for batch, (orig, labels) in enumerate(test):
    orig = Variable(orig.cuda())
    labels = Variable(labels.cuda())

    pred = model(orig)
    break

print(model)




