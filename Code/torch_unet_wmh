from PIL import Image
from glob import glob
import os
import nibabel as nib
import numpy as np
import pickle
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.transforms import functional as TF
from skimage.transform import resize, rescale, rotate
import torch

############### THIS FILE CONVERTS THE 3D MRI IMAGES INTO 2D SLICES FOR ALL 60 SUBJECTS       ################
############### 20% OF THE SLICES FOR EACH SUBJECT IS REMOVED AS THEY DO NOT CONTAIN LESIONS  ################
############### THIS IS DONE FOR BOTH FLAIR AND WMH-MASK IMAGES AND STORED IN PICKLE FILES    ################
############### NEXT THESE IMAGES TO BE CONVERTED TO SAME DIMESION; DTYPE AND INTENSITY       ################

def split_filename(filepath):
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext

def convert_to_slice(img_dir, out_dir, flair, mask, min_range = 0, max_range = 1, axis = 2):
    if flair:
        fns = glob(os.path.join(img_dir, 'FLAIR.nii*'))
    if mask:
        fns = glob(os.path.join(img_dir, 'wmh.nii*'))

    for fn in fns:
        _, base, ext = split_filename(fn)
        img = nib.load(fn).get_data().astype(np.float32).squeeze()
        if img.ndim != 3:
            print(f'Only 3D data supported. File {base}{ext} has dimension {img.ndim}. Skipping.')
            continue
        start = int(min_range * img.shape[axis])
        end = int(max_range * img.shape[axis])
        for i in range(start, end):
            I = Image.fromarray(img[:,:,i], mode='F')
            I.save(os.path.join(out_dir, f'{base}_{i:04}.tif'))

        if flair:
            print('FLAIR slices created ', out_dir)
        if mask:
            print('MASK slices created ', out_dir)

def folder_to_convert():
    patient_num = 60

    for i in range(0, patient_num):
        if i < 20:
            IN_DIR = 'P:\\WMH\\Utrecht\\Utrecht'
            OUT_DIR = 'P:\\WMH\\'
            dirs = os.listdir(IN_DIR)
            a = dirs.__getitem__(i)
            path = IN_DIR + '\\' + a + '\\pre'
            img_dir = os.path.join(path)

            mask_path = IN_DIR + '\\' + a
            mask_dir = os.path.join(mask_path)

            out_dir = OUT_DIR + a
            os.mkdir(out_dir)
            print(out_dir + ' is created.')

            FLAIR_out_dir = out_dir+'\\FLAIR'
            os.mkdir(FLAIR_out_dir)

            MASK_out_dir = out_dir+'\\MASK'
            os.mkdir(MASK_out_dir)

            convert_to_slice(img_dir, FLAIR_out_dir, flair=True, mask=False, min_range=0.1, max_range=0.9)
            convert_to_slice(mask_dir, MASK_out_dir, flair= False, mask= True, min_range=0.1, max_range=0.9)

        elif (i>=20) and (i<40):
            IN_DIR = 'P:\\WMH\\Singapore\\Singapore'
            OUT_DIR = 'P:\\WMH\\'

            dirs = os.listdir(IN_DIR)
            a = dirs.__getitem__(i%20)
            path = IN_DIR + '\\' + a + '\\pre'
            img_dir = os.path.join(path)

            mask_path = IN_DIR + '\\' + a
            mask_dir = os.path.join(mask_path)

            out_dir = OUT_DIR + a
            os.mkdir(out_dir)
            print(out_dir + ' is created.')

            FLAIR_out_dir = out_dir + '\\FLAIR'
            os.mkdir(FLAIR_out_dir)

            MASK_out_dir = out_dir + '\\MASK'
            os.mkdir(MASK_out_dir)

            convert_to_slice(img_dir, FLAIR_out_dir, flair=True, mask=False, min_range=0.1, max_range=0.9)
            convert_to_slice(mask_dir, MASK_out_dir, flair=False, mask=True, min_range=0.1, max_range=0.9)

        else:
            IN_DIR = 'P:\\WMH\\Amsterdam_GE3T\\GE3T'
            OUT_DIR = 'P:\\WMH\\'

            dirs = os.listdir(IN_DIR)
            a = dirs.__getitem__(i%40)
            path = IN_DIR + '\\' + a + '\\pre'
            img_dir = os.path.join(path)

            mask_path = IN_DIR + '\\' + a
            mask_dir = os.path.join(mask_path)

            out_dir = OUT_DIR + a
            os.mkdir(out_dir)
            print(out_dir + ' is created.')

            FLAIR_out_dir = out_dir + '\\FLAIR'
            os.mkdir(FLAIR_out_dir)

            MASK_out_dir = out_dir + '\\MASK'
            os.mkdir(MASK_out_dir)

            convert_to_slice(img_dir, FLAIR_out_dir, flair=True, mask=False, min_range=0.1, max_range=0.9)
            convert_to_slice(mask_dir, MASK_out_dir, flair=False, mask=True, min_range=0.1, max_range=0.9)


def prepare_train_data():
    HOME_DIR = 'P:\\WMH\\Train'
    files_list = os.listdir(HOME_DIR)

    patient_num = 60

    X = []
    y = []
    #train_data = []

    for i in range(0,patient_num):
        file = files_list.__getitem__(i)
        X_path = HOME_DIR + '\\' + file + '\\FLAIR'
        flair_imgs = os.listdir(X_path)

        Y_path = HOME_DIR + '\\' + file + '\\MASK'
        mask_imgs = os.listdir(Y_path)
    #
    #     for img_count in range(0, len(flair_imgs)):
    #         flair_item = flair_imgs.__getitem__(img_count)
    #         mask_item = mask_imgs.__getitem__(img_count)
    #
    #         flair_path = X_path + '\\' + flair_item
    #         mask_path = Y_path + '\\' + mask_item
    #
    #         flair = Image.open(flair_path)
    #         mask = Image.open(mask_path)
    #
    #         flair = np.array(flair)
    #         mask = np.array(mask)
    #
    #         train_data.append([flair,mask])
    #
    # pickle_out = open("train_data.pickle", 'wb')
    # pickle.dump(train_data, pickle_out)
    # pickle_out.close()

        for img in flair_imgs:
            path = X_path + '\\' + img
            #flair = Image.open(path)
            #flair = Image.open(path).resize((200,200))
            #flair = np.array(flair)
            X.append(path)

        for img in mask_imgs:
            path = Y_path + '\\' + img
            #mask = Image.open(path)
            #mask = Image.open(path).resize((200,200))
            #mask = np.array(mask)
            y.append(path)

    pickle_out = open("X.pickle", 'wb')
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("Y.pickle", 'wb')
    pickle.dump(y, pickle_out)
    pickle_out.close()

# class Transform():
#
#     def __init__(self, images, masks):
#         self.images = images
#         self.masks = masks
#         self.tensor_images = []
#         self.tensor_mask = []
#
#     def transform(self, image, mask):
#
#
#         resize = transforms.Resize(size=(200, 200))
#         image_resize = resize(image)
#         mask_resize = resize(mask)
#
#         affine = transforms.RandomAffine(degrees=[-15,15], scale=[0.9, 1.1], shear=[-18,18])
#         image_affine = affine(image_resize)
#         mask_affine = affine(mask_resize)
#
#         return transforms.ToTensor(image_affine),transforms.ToTensor(mask_affine)
#
#     def resize(self, image, mask):
#         # Resize
#         resize = transforms.Resize(size=(200,200))
#         image_resize = resize(image)
#         mask_resize = resize(mask)
#
#         return image_resize, mask_resize
#
#     def scale(self, image, mask):
#         # Scale
#         scale = transforms.RandomAffine(degrees=0, scale=[0.9,1.1])
#         image_scale = scale(image)
#         mask_scale = scale(mask)
#
#         image_tensor_scale = transforms.ToTensor(image_scale)
#         mask_tensor_scale = transforms.ToTensor(mask_scale)
#
#         return image_tensor_scale, mask_tensor_scale
#
#     def rotate(self, image, mask):
#         # Rotation
#         rotate = transforms.RandomRotation(15)
#         image_rotate = rotate(image)
#         mask_rotate = rotate(mask)
#
#         image_tensor_rotate = transforms.ToTensor(image_rotate)
#         mask_tensor_rotate = transforms.ToTensor(mask_rotate)
#
#         return image_tensor_rotate, mask_tensor_rotate
#
#
#     def get_item(self):
#         for index in range(0, len(self.images)):
#             image = self.images[index]
#             mask = self.masks[index]
#
#             image = Image.open(image)
#             mask = Image.open(mask)
#
#             im_re, ma_re = self.resize(image, mask)
#
#             # self.tensor_images.append(tensor_image)
#             # self.tensor_mask.append(tensor_mask)
#
#             # self.tensor_images.append(transforms.ToTensor(im_re))
#             # self.tensor_mask.append(transforms.ToTensor(ma_re))
#             #
#             # im_ro, ma_ro = self.rotate(im_re, ma_re)
#             #
#             # self.tensor_images.append(im_ro)
#             # self.tensor_mask.append(ma_ro)
#             #
#             # im_sc, ma_sc = self.scale(im_re, ma_re)
#             #
#             # self.tensor_images.append(im_sc)
#             # self.tensor_mask.append(ma_sc)
#
#         return self.tensor_images, self.tensor_mask
#
#
#     def __len__(self):
#         return len(self.image_paths)

class GenericImageDataset():

    def __init__(self, X, y, transformations):
        self.X = X
        self.y = y
        self.transformations = transformations

    def __getitem__(self, index):
        # actual_data = []
        # label = []

        #for index in range(0, len(self.X)):
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
    pickle_in = open('X.pickle', 'rb')
    X_path = pickle.load(pickle_in)

    pickle_in = open('y.pickle', 'rb')
    y_path = pickle.load(pickle_in)

    # transformations = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((200,200)),
    #     transforms.RandomAffine(degrees=[-15, 15], scale=[0.9, 1.1], shear=[-18, 18]),
    #     transforms.ToTensor()
    #     #transforms.Normalize([0.5], [0.5])
    # ])

    ## RESIZE
    trans1 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((200,200)),
        #transforms.RandomAffine(degrees=[-15, 15], scale=[0.9, 1.1], shear=[-18, 18]),
        transforms.ToTensor()
        #transforms.Normalize([0.5], [0.5])
    ])

    ## RESIZE AND ROTATE
    trans2 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((200,200)),
        transforms.RandomAffine(degrees=[-15, 15]),
        transforms.ToTensor()
        #transforms.Normalize([0.5], [0.5])
    ])

    ## RESIZE AND SCALE
    trans3 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((200,200)),
        transforms.RandomAffine(degrees=0, scale=[0.9, 1.1]),
        transforms.ToTensor()
        #transforms.Normalize([0.5], [0.5])
    ])

    ## RESIZE AND SHEAR
    trans4 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((200,200)),
        transforms.RandomAffine(degrees=0, shear=[-18, 18]),
        transforms.ToTensor()
        #transforms.Normalize([0.5], [0.5])
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

    # image, label = train_data[40]
    # print(image.shape)
    #
    # image, label = train_data[400]
    # print(image.shape)
    pickle_out = open("tensor.pickle", 'wb')
    pickle.dump(train_data, pickle_out)
    pickle_out.close()

#prepare_train_data()
#data_to_dim_shape()

class FullTrainningDataset(torch.utils.data.Dataset):
    def __init__(self, full_ds, offset, length):
        self.full_ds = full_ds
        self.offset = offset
        self.length = length
        assert len(full_ds) >= offset + length, Exception("Parent Dataset not long enough")
        super(FullTrainningDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.full_ds[i + self.offset]


validationRatio = 0.20


def trainTestSplit(dataset, val_share=validationRatio):
    val_offset = int(len(dataset) * (1 - val_share))
    return FullTrainningDataset(dataset, 0, val_offset), FullTrainningDataset(dataset, val_offset,
                                                                              len(dataset) - val_offset)

pickle_in = open('tensor.pickle','rb')
dset_train = pickle.load(pickle_in)

train_ds, val_ds = trainTestSplit(dset_train)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=10, shuffle=False, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=10, shuffle=False, num_workers=0)


import torch
import torch.nn as nn

class CleanU_Net(nn.Module):

    def __init__(self):

        super(CleanU_Net, self).__init__()

        # Conv block 1 - Down 1
        self.conv1_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),

        )
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 2 - Down 2
        self.conv2_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=96,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
        )
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 3 - Down 3
        self.conv3_block = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
        )
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 4 - Down 4
        self.conv4_block = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
        )
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 5 - Down 5
        self.conv5_block = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
        )

        # Up 1
        self.up_1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2)

        # Up Conv block 1
        self.conv_up_1 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=256,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
        )

        # Up 2
        self.up_2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2)

        # Up Conv block 2
        self.conv_up_2 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=128,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
        )

        # Up 3
        self.up_3 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2)

        # Up Conv block 3
        self.conv_up_3 = nn.Sequential(
            nn.Conv2d(in_channels=224, out_channels=96,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=96,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
        )

        # Up 4
        self.up_4 = nn.ConvTranspose2d(in_channels=96, out_channels=96, kernel_size=2, stride=2)

        # Up Conv block 4
        self.conv_up_4 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=64,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(3)
        )

        # Final output
        self.conv_final = nn.Conv2d(in_channels=64, out_channels=1,
                                    kernel_size=1, padding=(1,1), stride=1)


    def forward(self, x):
        # print('input', x.shape)

        # Down 1
        x = self.conv1_block(x)
        # print('after conv1', x.shape)
        conv1_out = x  # Save out1
        conv1_dim = x.shape[2]
        x = self.max1(x)
        # print('before conv2', x.shape)

        # Down 2
        x = self.conv2_block(x)
        # print('after conv2', x.shape)
        conv2_out = x
        conv2_dim = x.shape[2]
        x = self.max2(x)
        # print('before conv3', x.shape)

        # Down 3
        x = self.conv3_block(x)
        # print('after conv3', x.shape)
        conv3_out = x
        conv3_dim = x.shape[2]
        x = self.max3(x)
        # print('before conv4', x.shape)

        # Down 4
        x = self.conv4_block(x)
        # print('after conv5', x.shape)
        conv4_out = x
        conv4_dim = x.shape[2]
        x = self.max4(x)

        # Midpoint
        x = self.conv5_block(x)

        # Up 1
        x = self.up_1(x)
        # print('up_1', x.shape)
        lower = int((conv4_dim - x.shape[2])/2)
        upper = int(conv4_dim - lower)

        conv4_out_modified = conv4_out[:, :, 0:x.shape[2], 0:x.shape[3]]
        x = torch.cat([x, conv4_out_modified], dim=1)


        # print('after cat_1', x.shape)
        x = self.conv_up_1(x)
        # print('after conv_1', x.shape)

        # Up 2
        x = self.up_2(x)
        # print('up_2', x.shape)
        lower = int((conv3_dim - x.shape[2]) / 2)
        upper = int(conv3_dim - lower)
        conv3_out_modified = conv3_out[:, :, 0:x.shape[2], 0:x.shape[3]]
        x = torch.cat([x, conv3_out_modified], dim=1)
        # print('after cat_2', x.shape)
        x = self.conv_up_2(x)
        # print('after conv_2', x.shape)

        # Up 3
        x = self.up_3(x)
        # print('up_3', x.shape)
        lower = int((conv2_dim - x.shape[2]) / 2)
        upper = int(conv2_dim - lower)
        conv2_out_modified = conv2_out[:, :, 0:x.shape[2], 0:x.shape[3]]
        x = torch.cat([x, conv2_out_modified], dim=1)
        # print('after cat_3', x.shape)
        x = self.conv_up_3(x)
        # print('after conv_3', x.shape)

        # Up 4
        x = self.up_4(x)
        # print('up_4', x.shape)
        lower = int((conv1_dim - x.shape[2]) / 2)
        upper = int(conv1_dim - lower)
        conv1_out_modified = conv1_out[:, :, 0:x.shape[2], 0:x.shape[3]]
        x = torch.cat([x, conv1_out_modified], dim=1)
        # print('after cat_4', x.shape)
        x = self.conv_up_4(x)
        # print('after conv_4', x.shape)

        # Final output
        x = self.conv_final(x)

        return x

def train_model(train_loader):
    model = CleanU_Net()

    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).cuda()

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Optimizerd
    #optimizer = torch.optim.RMSprop(model.module.parameters(), lr=0.0002)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0002)

    from torch.autograd import Variable

    epochs = 1
    device = torch.device('cuda')

    from torch.utils.tensorboard import SummaryWriter

    model.train()
    for epoch in range(epochs):

        for batch, (images, masks) in enumerate(train_loader):

            print('Executing epoch: ', epoch, ' and batch: ', batch)

            images = images.to(device=device, dtype=torch.float32)
            mask_type = torch.float32
            masks = masks.to(device=device, dtype=mask_type)

            masks_pred = model(images)
            loss = criterion(masks_pred, masks)
            print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    PATH = 'P:\\WMH\\EPOCH\\model.pth'
    torch.save(model.state_dict(), PATH)


#train_model(train_loader)

from collections import OrderedDict

def validate_data(val_loader):
    PATH = 'P:\\WMH\\EPOCH\\model.pth'
    model = CleanU_Net()
    #model.load_state_dict(torch.load(PATH))
    state_dict = torch.load(PATH)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    model.cuda()
    print(model)

    device = torch.device('cuda')
    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    for batch, (images, masks) in enumerate(val_loader):
        print('Executing batch: ', batch)
        images = images.to(device=device, dtype=torch.float32)
        mask_type = torch.float32
        masks = masks.to(device=device, dtype=mask_type)

        masks_pred = model(images)
        loss = criterion(masks_pred, masks)
        print(loss.item())

validate_data(val_loader)
