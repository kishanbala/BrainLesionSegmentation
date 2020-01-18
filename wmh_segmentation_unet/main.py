import prepare_axial_slices
import store_image_dirs
import data_format_augment
import train_test_split
import train_network
import test_model


import pickle
import matplotlib.pyplot as plt
import numpy as np

# This method 'folder_to_convert' should be called only once
#prepare_axial_slices.folder_to_convert()

# This method collects all the image paths - done only once to generate X & y pickle files
#store_image_dirs.prepare_train_data()

# Transforms the data to tensor after formatting & augmentation - done only once to generate tensor.pickle file
data_format_augment.data_to_dim_shape()


# ##SANITY CHECK - OKAY - QUALITY OF TRANSFORMED TENSOR IMAGES SEEMS GOOD
pickle_in = open('tensor.pickle','rb')
dset_train = pickle.load(pickle_in)
print(len(dset_train))

image, mask = dset_train[10]
image_ = np.squeeze(image)
plt.imshow(image_,cmap='gray')
plt.title('Image')
plt.show()

mask_ = np.squeeze(mask)
plt.imshow(mask_,cmap='gray')
plt.title('Mask')
plt.show()

# Generate training and validation datasets
#train_loader, val_loader = train_test_split.create_train_val_dset()

# Train the network and save the best model in 'model_best.pth.tar'
#train_network.train_dataset(train_loader)

# # Test the trained model on validation dataset
#test_model.test(val_loader)

