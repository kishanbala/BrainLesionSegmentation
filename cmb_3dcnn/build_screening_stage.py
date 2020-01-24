import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=64,
                               kernel_size=(3, 5, 5), stride=(1, 1, 1), bias=False)
        self.max1 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=False)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1, 3, 3), stride=(1, 1, 1), bias=False)

        #self.fc1 = nn.Linear(64,150)
        self.fc1 = nn.Conv3d(in_channels=64, out_channels=150, kernel_size=(2, 2, 2), stride=(1, 1, 1), bias=False)
        #self.fc2 = nn.Linear(150, 2)
        self.fc2 = nn.Conv3d(in_channels=150, out_channels=2, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)

    def forward(self, x):
        #print('Input: ',x.shape)
        x = self.conv1(x)
        x = F.relu(x)

        print('After conv 1: ',x.shape)

        x = self.max1(x)
        print('After max pool 1: ', x.shape)

        x = self.conv2(x)
        x = F.relu(x)

        print('After conv 2: ',x.shape)

        x = self.conv3(x)
        x = F.relu(x)
        print('After conv 3: ', x.shape)

        # x = x.reshape(x.size(0), -1)
        # #print('After reshape :', x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        print('After full conv 1: ', x.shape)

        x = self.fc2(x)
        x = F.relu(x)
        print('After full conv 2: ', x.shape)
        return x

import train_val_split
import torch.optim as optim
from torch.autograd import Variable



train_dset, val_dset, test_dset = train_val_split.train_test_split()

model = CNN().cuda()
model.train()

for i, (images, labels) in enumerate(train_dset):
    images = Variable(images.cuda())
    labels = Variable(labels.cuda())

    outputs = model(images)
    break