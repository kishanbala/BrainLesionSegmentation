import torch
import torch.nn as nn
import os
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import matplotlib
from matplotlib import pyplot as plt
from torch.autograd import Variable
import glob
import torch.optim as optim
from PIL import Image
import nibabel as nib

IMAGE_PATH = glob.glob("E:\\CMB\\nii\\*.nii")
MASK_PATH = glob.glob("E:\\CMB\\ground_truth\\*.nii")

os.chdir("E:\\cerebral")


class CMB_DataSet(Dataset):
    def __init__(self,image_paths,mask_paths,transforms):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __getitem__(self,index):
        image = nib.load(self.image_paths[index])
        mask = nib.load(self.mask_paths[index])
        image = np.asarray(image.dataobj).astype(np.float32)
        # image = np.expand_dims(image, axis=0)
        t_image = self.transforms((image))
        # t_mask = torch.from_numpy(mask) #BCELoss
        mask = np.asarray(mask.dataobj).astype(np.float32)
        # mask = np.expand_dims(mask, axis=0)
        t_mask = self.transforms(mask)
        return t_image,t_mask

    def __len__(self):
        return len(self.image_paths)


transformations = transforms.ToTensor()

master = CMB_DataSet(IMAGE_PATH,MASK_PATH,transforms=transformations)

for i in range(0,master.__len__()):
    image_tensor,mask_tensor = master.__getitem__(i)

length = master.__len__()

n_train = int(length * 0.8)
n_test = int(length * 0.2)
idx = list(range(length))
train_idx = idx[: n_train]
test_idx = idx[n_train:]

train_set = data.Subset(master,train_idx)
test_set = data.Subset(master,test_idx)

train_loader = torch.utils.data.DataLoader(train_set,batch_size=2,shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=1,shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1,out_channels=32,kernel_size=(3,3,3),padding=0,bias=False)
        self.conv2 = nn.Conv3d(in_channels=32,out_channels=64,kernel_size=(1,4,4),padding=0,bias=False)
        self.conv3 = nn.Conv3d(in_channels=64,out_channels=128,kernel_size=(1,5,5),padding=0,bias=False)
        self.fc1 = nn.Linear(128 * 1936,2)
        self.fc2 = nn.Linear(2,2)

    def forward(self,x):
        print('Input: ',x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        print('After conv 1: ',x.shape)
        x = F.MaxPool3d(kernel_size=(1,4,4),stride=(2,2,2))

        x = self.conv2(x)
        x = F.ReLU(x)
        x = F.MaxPool3d(kernel_size=(1,4,4),stride=(2,2,2))

        print('After conv 2: ',x.shape)

        x = self.conv3(x)
        x = F.ReLU()
        x = F.MaxPool3d(kernel_size=(1,5,5),stride=(2,2,2))
        x = x.reshape(x.size(0),-1)
        print('After conv 3: ',x.shape)

        x = F.relu(self.fc1(x))
        print('After full conv 1: ',x.shape)
        x = F.relu(self.fc2(x))
        print('After full conv 2: ',x.shape)
        return x


def train():
    model = CNN()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    epochs = 1

    for batch,(images,masks) in enumerate(train_loader):
        images = images.unsqueeze(1)
        # masks = masks.to(dtype=torch.long)
        masks = masks.unsqueeze(1)
        print(images.shape)
        print(masks.shape)
        output = model(images)
        print(output.shape)
        break


train()
