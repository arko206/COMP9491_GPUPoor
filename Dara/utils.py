import numpy as np
import torch
from torch.utils.data import Dataset


def add_noise(x, mean=0.0, std=0.05):
    return x + torch.normal(mean, std, x.shape).float()

def load_pems(data_path):
    return np.load(data_path).get('data').transpose(1, 2, 0)[:, :1]

def get_train_test_splits(data, train_size=0.6, test_size=0.2):
    assert train_size + test_size < 1.0
    # input: [B, C, T]
    seq_len = data.shape[-1]
    n_train = round(seq_len * 0.6)
    n_test = round(seq_len * 0.2)
    n_val = seq_len - n_train - n_test
    data_train = data[:, :, :n_train]
    data_val = data[:, :, n_train:n_train+n_val]
    data_test = data[:, :, -n_test:]
    return (
        data_train,
        data_val,
        data_test,
    )

def split_time_series(X, l):
    ''' X: input time series, [B, C, T]
        l: sub sequence length (after split)
    '''
    chunks = X.shape[-1] // l
    return np.concatenate([X[:, :, l*i:l*(i+1)] for i in range(chunks)])


class TSTrainDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]