import pickle
import torch.utils.data as data
import torch


def train_test_split():
    pickle_in = open('dataset.pickle', 'rb')
    dset = pickle.load(pickle_in)
    length = len(dset)

    n_train = int(length * 0.6)
    n_test = int(length * 0.2)
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


