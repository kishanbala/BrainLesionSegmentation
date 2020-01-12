# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 16:48:32 2020

@author: Bala
"""

import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import transforms
from CMB import dice_loss


IMAGE_PATH = glob.glob("E:\\CerebralMicroBleeds\\Images\\*.tif")
MASK_PATH = glob.glob("E:\\CerebralMicroBleeds\\mask\\*.tif")

os.chdir("E:\\CerebralMicroBleeds\\")

#print(os.getcwd())

class CMB_DataSet(Dataset):
    def __init__(self, image_paths, mask_paths, transforms):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
        
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.mask_paths[index])
        t_image = self.transforms(image)
        t_mask = self.transforms(mask)
        return t_image, t_mask
    
    
    def __len__(self):
        return len(self.image_paths)
    

transformations = transforms.ToTensor()

master = CMB_DataSet(IMAGE_PATH, MASK_PATH, transforms = transformations)
length = master.__len__()

n_train = int(length * 0.8)
n_test = int(length * 0.1)
idx = list(range(length))
train_idx = idx[: n_train]
test_idx = idx[n_train : ]


train_set = data.Subset(master, train_idx)
test_set = data.Subset(master, test_idx)


train_loader = torch.utils.data.DataLoader(train_set, batch_size = 2, shuffle = False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = False)


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2,2), padding = (4,4),  stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64,  kernel_size=(2,2), padding = (4,4), stride=(1,1))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(2,2), padding = (4,4), stride=(1,1))
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=1024,  kernel_size=(2,2), padding = (4,4), stride=(1,1))
        self.conv5 = nn.Conv2d(in_channels=1024, out_channels=4096,  kernel_size=(2,2), padding = (4,4), stride=(1,1))

        self.fc1 = nn.Linear(in_features=4096*18*18,  out_features=18*18)
        self.fc2 = nn.Linear(in_features=18*18, out_features=372*372)
    
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        #print(x.shape)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        #print(x.shape)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        #print(x.shape)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        #print(x.shape)
        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        #print(x.shape)
        ########################## feed output to fully connected layer #############
        # vectorizing the input matrix
        
        x = x.reshape(x.size(0), -1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        #print(x.shape)
        #x = x.reshape(x.size(0),-1)
        x = self.fc2(x)
        return x
    

        
def train_model(train_loader):
    
    model = Net(2)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    criterion = dice_loss()
    
    model.train()


    epochs = 1
    
    for epoch in range(epochs):
       
        print("Executing Epoch:", epoch)
        
        print("-" * 10)
        
        print("Iterating through data")
        
        current_loss = 0.0
        
            
        for batch, (images, masks) in enumerate(train_loader):
            images = images
            #print(images.shape)
            masks = masks.to(dtype = torch.long)
            masks = masks.reshape(masks.size(0), -1)
            #print(masks.shape)
            output = model(images)
            #print(output.shape)
            loss = criterion(output, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            current_loss += loss.item()* images.size(1)
            
            epoch_loss = current_loss / output.shape[0]
            
            print("Batch:", batch, "Train Loss: {:.4f}".format(epoch_loss)
            
        
        PATH = "E:\\CerebralMicroBleeds\\model\\CMB_Trained.pth"
        torch.save(model.state_dict(), PATH)


 
  train_model(train_loader)