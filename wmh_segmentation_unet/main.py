import prepare_axial_slices
import store_image_dirs
import data_format_augment
import train_test_split
import build_unet_architecture
import loss
import train_network
import save_model


import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tensorboardX import SummaryWriter


import torch
import torch.optim as optim
import time
import copy

# This method 'folder_to_convert' should be called only once
#prepare_axial_slices.folder_to_convert()

# This method collects all the image paths - done only once to generate X & y pickle files
#store_image_dirs.prepare_train_data()

# Transforms the data to tensor after formatting & augmentation - done only once to generate tensor.pickle file
#data_format_augment.data_to_dim_shape()

#SANITY CHECK - OKAY - QUALITY OF TRANSFORMED TENSOR IMAGES SEEMS GOOD
# pickle_in = open('tensor.pickle','rb')
# dset_train = pickle.load(pickle_in)
# print(len(dset_train))
#
# image, mask = dset_train[11000]
# image_ = np.squeeze(image)
# plt.imshow(image_)
# plt.title('Image')
# plt.show()
#
# mask_ = np.squeeze(mask)
# plt.imshow(mask_)
# plt.title('Mask')
# plt.show()

# Generate training and validation datasets
train_loader, val_loader = train_test_split.create_train_val_dset()

model = build_unet_architecture.CleanU_Net().cuda()
criterion = loss.BCELoss2d().cuda()
optimizer = optim.SGD(model.parameters(),
                      weight_decay=1e-4,
                      lr=1e-4,
                      momentum=0.9,
                      nesterov=True)

# set somo global vars
experiment = "your_experiment"
logger = SummaryWriter(comment=experiment)
best_loss = 0

# run the training loop
num_epochs = 20
for epoch in range(0, num_epochs):
    # train for one epoch
    curr_loss = train_network.train(train_loader, model, criterion, epoch, num_epochs)

    # store best loss and save a model checkpoint
    is_best = curr_loss < best_loss
    best_loss = max(curr_loss, best_loss)
    save_model.save_checkpoint({
        'epoch': epoch + 1,
        'arch': experiment,
        'state_dict': model.state_dict(),
        'best_prec1': best_loss,
        'optimizer': optimizer.state_dict(),
    }, is_best)

logger.close()


