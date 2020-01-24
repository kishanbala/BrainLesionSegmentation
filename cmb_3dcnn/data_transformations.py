from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from torchvision import transforms
import pickle
from torch import Tensor

class CMB_DataSet(Dataset):
    def __init__(self,image_paths,mask_paths,transforms):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __getitem__(self,index):
        #image = nib.load(self.image_paths[index])
        #mask = nib.load(self.mask_paths[index])

        image = nib.load(self.image_paths[index]).get_data().astype(np.float32)
        mask = nib.load(self.mask_paths[index]).get_data().astype(np.float32)

        #image = np.asarray(image.dataobj).astype(np.float32)
        # image = np.expand_dims(image, axis=0)
        t_image = self.transforms((image))
        # t_mask = torch.from_numpy(mask) #BCELoss
        #mask = np.asarray(mask.dataobj).astype(np.float32)
        # mask = np.expand_dims(mask, axis=0)
        t_mask = self.transforms(mask)
        return t_image,t_mask

    def __len__(self):
        return len(self.image_paths)

pickle_in = open('input_paths.pickle', 'rb')
IMAGE_PATH = pickle.load(pickle_in)

pickle_in = open('gt_paths.pickle', 'rb')
MASK_PATH = pickle.load(pickle_in)


transformations = transforms.ToTensor()

master = CMB_DataSet(IMAGE_PATH,MASK_PATH,transforms=transformations)


dataset = []

for i in range(0,master.__len__()):
    image_tensor,mask_tensor = master.__getitem__(i)
    image_tensor = np.expand_dims(image_tensor, axis=0)
    mask_tensor = np.expand_dims(mask_tensor, axis=0)

    image_tensor = Tensor(image_tensor)
    mask_tensor = Tensor(mask_tensor)

    # print(image_tensor.shape)
    # print(image_tensor.min(), image_tensor.max())
    # print(mask_tensor.shape)
    # print(type(image_tensor))
    # print(type(mask_tensor))

    print('Processing ',i, ' images')

    if(image_tensor.max() > 0.0):
        dataset.append([image_tensor, mask_tensor])

# print(len(dataset))
#
print('Saving to pickle')
pickle_out = open("dataset.pickle", 'wb')
pickle.dump(dataset, pickle_out)
pickle_out.close()