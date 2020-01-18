
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# original = Image.open('C:\\Users\\haris\\PycharmProjects\\wmh_segmentation_unet\\validation\\input_image_274\\original_image.png')
# segment = Image.open('C:\\Users\\haris\\PycharmProjects\\wmh_segmentation_unet\\validation\\input_image_274\\original_seg_output.png')
# predicted = Image.open('C:\\Users\\haris\\PycharmProjects\\wmh_segmentation_unet\\validation\\input_image_274\\predicted_seg_output.png')
#
# original = np.array(original)
# segment = np.array(segment)
# predicted = np.array(predicted)
#
# original = np.squeeze(original)
# segment = np.squeeze(segment)
# predicted = np.squeeze(predicted)
#
# plt.figure(figsize=(15,3))
# plt.subplot(131)
# plt.imshow(original)
# plt.title('Image')
# plt.subplot(132)
# plt.imshow(segment,cmap='gray')
# plt.title('Ground Truth')
# plt.subplot(133)
# plt.imshow(predicted,cmap='gray')
# plt.title('Mask')
# plt.show()

import matplotlib.pyplot as plt
import pickle

pickle_in = open('tensor.pickle','rb')
dset_train = pickle.load(pickle_in)
print(len(dset_train))

image, mask = dset_train[0]
image_ = np.squeeze(image)

image, mask = dset_train[2880]
rotated_ = np.squeeze(image)

image, mask = dset_train[5760]
scaled_ = np.squeeze(image)

image, mask = dset_train[8640]
sheared_ = np.squeeze(image)


plt.figure(figsize=(15,3))
# plt.subplot(131)
# plt.imshow(image_, cmap='gray')
# plt.title('Image')
plt.subplot(131)
plt.imshow(rotated_,cmap='gray')
plt.title('Rotation')
plt.subplot(132)
plt.imshow(scaled_,cmap='gray')
plt.title('Scaling')
plt.subplot(133)
plt.imshow(scaled_,cmap='gray')
plt.title('Shearing')
plt.show()

