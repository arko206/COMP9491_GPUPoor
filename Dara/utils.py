import numpy as np
import torch
from torch.utils.data import Dataset


def add_noise(x, mean=0.0, std=0.05):
    return x + torch.normal(mean, std, x.shape).float()

def load_pems(data_path):
    return np.load(data_path).get('data').transpose(1, 2, 0)[:, :1]

def get_crops(x, overlap_size):
    ''' input: [B, C, T]
    '''
    seq_len = x.size(2)
    crop_len = int(seq_len * overlap_size + seq_len * (1 - overlap_size)/2)
    x1_start = 0
    x1_end = x1_start + crop_len
    x2_start = seq_len - crop_len
    x2_end = seq_len
    return (
        x[:, :, x1_start:x1_end],
        x[:, :, x2_start:x2_end],
        int(seq_len * overlap_size),
    )

def get_train_test_splits(data, train_size=0.6, test_size=0.2):
    assert train_size + test_size < 1.0
    # input: [B, C, T]
    seq_len = data.shape[-1]
    n_train = round(seq_len * train_size)
    n_test = round(seq_len * test_size)
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
    
def create_lag_features(ts, hist_timesteps, forecast_steps, skip):
    ''' ts: [B, C, T]
    '''
    X, y = [], []
    for i in range(0, ts.shape[-1] - hist_timesteps - forecast_steps + 1, skip):
        X.append(ts[:, :, i:i+hist_timesteps])
        y.append(ts[:, :, i+hist_timesteps:i+hist_timesteps+forecast_steps])
    return np.array(X), np.array(y).squeeze()