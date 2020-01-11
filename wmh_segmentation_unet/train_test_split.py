import pickle
import torch

class FullTrainningDataset(torch.utils.data.Dataset):
    '''
    Performs indexing on whole dataset to split them as train & validation datasets
    '''
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

## SET VALIDATION SET SIZE & BATCH SIZE
validationRatio = 0.20
batch_size = 10

def trainTestSplit(dataset, val_share=validationRatio):
    '''

    :param dataset: Complete dataset in X,y pair after formatting & augmenting
    :param val_share: Validation dataset size
    :return: Train and test datasets
    '''
    val_offset = int(len(dataset) * (1 - val_share))
    return FullTrainningDataset(dataset, 0, val_offset), FullTrainningDataset(dataset, val_offset,
                                                                              len(dataset) - val_offset)
def create_train_val_dset():
    pickle_in = open('tensor.pickle','rb')
    dset_train = pickle.load(pickle_in)

    train_ds, val_ds = trainTestSplit(dset_train)

    ## USE THESE FOR TRAINING & EVALUATING MODEL
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader