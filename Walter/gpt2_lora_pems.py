import argparse
import os
import random
import gc
import math

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm
from sklearn import preprocessing
from tqdm.auto import tqdm
from datetime import datetime
from llm import CustomFeatureExtractor, apply_lora

import torch
torch.cuda.empty_cache()


import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType


"""
Input dimension of original dataset is:
M * N * 3
M means the total timesteps
N means the number of sensors
3 means traffic flow, traffic speed and occupancy
"""


def set_env(seed):
    # Set available CUDA devices
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


def get_parameters():
    parser = argparse.ArgumentParser(description='TPLLM')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='PEMS07', choices=['PEMS04', 'PEMS08', 'PEMS07'])
    parser.add_argument('--F_channel',type=int, default=32)
    parser.add_argument('--n_step',type=int, default=6)
    parser.add_argument('--n_his', type=int, default=12)
    parser.add_argument('--n_pred', type=int, default=12, help='the number of time interval for predcition, default as 3')
    parser.add_argument('--device', type=str, default='cuda:7')
    parser.add_argument('--context_dim', type=int, default=128)
    parser.add_argument('--n_vertex', type=int, default=0)
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv', choices=['cheb_graph_conv', 'graph_conv'])
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.001, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100, help='epochs, default as 1000')
    parser.add_argument('--opt', type=str, default='lion', choices=['adamw', 'lion', 'tiger'], help='optimizer, default as lion')
    parser.add_argument('--step_size', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--val_test_rate', type=float, default=0.2, choices=[0.15, 0.2],help='validation set and test set rate')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    # For stable experiment results
    set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device(args.device)
        torch.cuda.empty_cache() # Clean cache
    else:
        device = torch.device('cpu')
        gc.collect() # Clean cache
    
    return args, device


"""
directly return the whole data
choose the predict length by 
"""
class TPLLM_Dataset(Dataset):
    def __init__(self, x, y, adj): 
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        nonzero_indices = np.vstack(adj.nonzero())
        self.edge_index = torch.tensor(nonzero_indices, dtype=torch.long)
        # self.edge_weight = torch.tensor(adj[nonzero_indices[0], nonzero_indices[1]], dtype=torch.float)
        # print('--- check TPLLM dataset', self.edge_index.shape)
    
    def __getitem__(self, index):
        # print('check get item', self.x.shape)
        x_i = self.x[index].permute(1, 0)
        y_i = self.y[index].permute(1, 0)
        # data = Data(x=x_i, y=y_i, edge_index=self.edge_index, edge_attr=self.edge_weight)
        """
        Data put x, y,, edge_index into the shape for GCN
        """
        data = Data(x=x_i, y=y_i, edge_index=self.edge_index)
        # temporally only return data, when dealing with TCN, return x_i and y_i
        return data
    
    def __len__(self):
        return self.x.shape[0]


# def create_data_list(x_train, y_train, adj_matrix_coo):
#     data_list = []
#     for i in range(x_train.shape[0]):
#         x = x_train[i]  # (12, 307)
#         y = y_train[i]  # (3, 307)
#         # print('check inside', x.shape, y.shape)
#         edge_index = torch.tensor(np.vstack(adj_matrix_coo.nonzero()), dtype=torch.long)
#         edge_weight = torch.tensor(adj_matrix_coo[adj_matrix_coo.nonzero()], dtype=torch.float)
#         data = Data(x=x, y=y, edge_index=edge_index, edge_weight=edge_weight)
#         data_list.append(data)
#     return data_list


"""
model structure
temporary here, when becomes big, move it to another
model.py file

a long road: actually when after using Batch from torch_geometric
input shape don't need to be (batch_size, n_his, num_nodes)
edge_index shape don't need to be (batch_size, 2, num_edges)

it should be
input shape == (batch_size * num_nodes, n_his), in order to use conv layer
edge_index == (2, batch_size * num_edges)

"""
class Model_TPLLM(nn.Module):
    def __init__(self, args, edge_count, llm_in) -> None:
        super(Model_TPLLM, self).__init__()
        self.args = args
        self.en_gcn = GCN_layer(self.args, edge_count)
        self.en_tcn = TCN_layer(self.args)
        
        """
        GCN+TCN+LSTM
        """
        # self.llm = nn.LSTM(input_size=self.args.context_dim, hidden_size=self.args.context_dim, batch_first=False)
        """
        GCN+TCN+Transformer
        """
        # self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.args.context_dim, nhead=4, dim_feedforward=4*self.args.context_dim)
        # self.llm = nn.TransformerEncoder(self.transformer_layer, num_layers=3)
        self.llm = llm_in
        print('check hidden size', self.llm.config.hidden_size)
        self.fc_emb = nn.Linear(self.args.context_dim, self.llm.config.hidden_size)
        self.fc_llm_out = nn.Linear(self.llm.config.hidden_size, self.args.context_dim)
        
        self.proj1 = nn.Linear(self.args.F_channel * self.args.n_his, self.args.context_dim)
        self.proj2 = nn.Sequential(
            nn.Linear(self.args.context_dim, self.args.context_dim // 2),
            nn.ReLU(),
            nn.Linear(self.args.context_dim // 2, self.args.context_dim // 4),
            nn.ReLU(),
            nn.Linear(self.args.context_dim // 4, self.args.n_pred)
        )
        self.edge_count = edge_count
        
    def forward(self, data):
        """
        seq_len = n_vertex
        the number of sequence is equal to the number of vertex
        """
        x_gcn = self.en_gcn(data) # -- (seq_len * batch_size, F_channel, input_dim)
        # print('-- x_gcn shape', x_gcn.shape)
        x_tcn = self.en_tcn(data) # -- (seq_len * batch_size, F_channel, input_dim)
        # print('-- x_tcn shape', x_tcn.shape)
        x_in = x_gcn + x_tcn # -- (seq_len * batch_size, F_channel, input_dim
        # x_in = x_gcn
        x_llm_in = self.proj1(x_in.view(x_in.shape[0], -1)) # -- (seq_len * batch_size, context_dim)
        # print('check model inside:', x_llm_in.shape)
        """
        batch_size maybe smaller at the final epoch
        so use args.batch_size as the condition of .view() will cause some problems
        instead, use 'seq_len' as the condition of .view()
        """
        x_llm_in = x_llm_in.view(self.args.n_vertex, -1, self.args.context_dim) # -- (seq_len, batch_size, context_dim)
        # print('check x_llm_in shape:', x_llm_in.shape)
        """
        Deal with how to use the llm as the centre
        """
        inputs_embeds = self.fc_emb(x_llm_in)
        # x_llm_out = self.llm(input_embeds=input_embeds)['logits']
        """
        for phi-3 use
        """
        # transformer_outputs = self.llm.model(inputs_embeds=inputs_embeds, output_hidden_states=True)
        # logits = transformer_outputs.hidden_states[-1]
        # # print('check logit shape', logits.shape)
        # x_llm_out = self.fc_llm_out(logits)
        
        transformer_outputs = self.llm.transformer(inputs_embeds=inputs_embeds)
        logits = transformer_outputs.last_hidden_state
        x_llm_out = self.fc_llm_out(logits)
        # print('check x_llm_out shape', x_llm_out.shape)
        # print('check nodel inside after view', x_llm_in.shape)
        # print('-- x_llm_in shape', x_llm_in.shape)
        # x_llm_out = self.llm(x_llm_in) # -- (seq_len, batch_size, context_dim)
        # print('-- x_llm_out shape', x_llm_out.shape)
        x_out = self.proj2(x_llm_out) # -- (seq_len, batch_size, n_pred)
        # print('-- x_out shape', x_out.shape)
        x_out = x_out.view(-1, self.args.n_pred) # -- (seq_len * batch_size, n_predn )
        # print('-- x_out shape final', x_out.shape)
        return x_out


class TCN_layer(nn.Module):
    def __init__(self, args) -> None:
        super(TCN_layer, self).__init__()
        self.args = args
        self.tcn1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.tcn2 = nn.Conv1d(32, self.args.F_channel, kernel_size=3, padding=1)
        
    def forward(self, data):
        x = data.x.view(data.x.shape[0], 1, -1)
        x = F.relu(self.tcn1(x))
        x = F.relu(self.tcn2(x))
        return x

"""
input of GCN_layer is (b*N, 12)
the procedure is (b*N, 12) -> (b*N, 64/128/12*F) -> (b*N, F, 12)
"""
class GCN_layer(nn.Module):
    def __init__(self, args, edge_count):
        super(GCN_layer, self).__init__()
        self.args = args
        self.conv1 = GCNConv(args.n_his, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, args.n_his * args.F_channel)
        # self.n_his = n_his
        self.edge_count = edge_count
        
    def forward(self, data):
        # _, num_nodes = data.x.shape
        # x = data.x.view(-1, self.n_his, num_nodes)
        # edge_index = data.edge_index.view(-1, 2, self.edge_count)
        # x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # x = x.permute(0, 2, 1)
        # edge_index = edge_index.permute(0, 2, 1)
        # print('---- check x shape', data.x.shape, data.edge_index.shape)
        # x = self.conv1(x, data.edge_index, data.edge_weight)
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index)
        x = F.relu(x)
        x = self.conv3(x, data.edge_index)
        # x = self.conv2(x, data.edge_index, data.edge_weight)
        return x.view(x.shape[0], self.args.F_channel, self.args.n_his)



class Agent:
    def __init__(self) -> None:
        pass
    
    def train():
        pass


def collate_fn(batch, device):
    batch = Batch.from_data_list(batch)
    batch = batch.to(device)
    # print('check collate fn', batch.edge_index.shape)
    # if batch.edge_weight is None:
    #     print('---- Batch is None')
    return batch

"""
using sliding window to split samples
with a sliding step of 6
not 1 because it won't be necessary with too many overlapping
"""
# def data_transform(data, n_his, n_pred, device, step=3):
def data_transform(data, n_his, n_pred, step=3):
    # produce data slices for x_data and y_data
    # print('data transform', data.shape)
    n_vertex = data.shape[1]
    len_record = len(data)
    # print('len record', len_record)
    num = (len_record - n_his - n_pred) // step
    
    x = np.zeros([num, n_his, n_vertex])
    y = np.zeros([num, n_pred, n_vertex])
    
    for i in range(num):
        head = i * step
        tail = head + n_his
        x[i] = data[head: tail]#.reshape(1, n_his, n_vertex)
        y[i] = data[tail: tail+n_pred]

    return x, y


def data_prepare(args, device):

    data_folder = './data/' + args.dataset + '/'
    
    adj_path = data_folder + 'adj_matrix.npy'
    adj_matrix = np.load(adj_path)
    adj_matrix_coo = sp.coo_matrix(adj_matrix)
    adj_matrix_coo = adj_matrix_coo.tocsc()
    nonzero_indices = np.vstack(adj_matrix_coo.nonzero())
    # print('check adj matrix', adj_matrix_coo.shape)
    
    # gso = calc_gso(adj_matrix_coo, args.gso_type)
    # if args.graph_conv_type == 'cheb_graph_conv':
    #     gso = calc_chebynet_gso(gso)
    # gso = gso.toarray()
    
    n_vertex = adj_matrix.shape[0]
    
    """
    1. load the dataset
    """
    dataset_path = data_folder + 'dataset.npz'
    dataset = np.load(dataset_path)['data'][:,:,0]
    data_col = dataset.shape[0]
    
    """
    2. split the dataset into train/val/test
    """
    len_val = int(math.floor(data_col * args.val_test_rate))
    len_test = int(math.floor(data_col * args.val_test_rate))
    len_train = data_col - len_val - len_test
    trainset = dataset[:len_train,:]
    valset = dataset[len_train:len_train+len_val,:]
    testset = dataset[len_train+len_val:,:]
    
    """
    3. normalization, using StandardScaler
    """
    
    # print('check trainset shape before zscore', trainset.shape)
    zscore = preprocessing.StandardScaler()
    trainset = zscore.fit_transform(trainset)
    valset = zscore.fit_transform(valset)
    testset = zscore.fit_transform(testset)
    
    print('check after zsocre', trainset.shape, valset.shape, testset.shape)
    """
    4. using data_transform to split data into samples
    """
    # x_train, y_train = data_transform(trainset, args.n_his, args.n_pred, device, step=3)
    # x_val, y_val = data_transform(valset, args.n_his, args.n_pred, device, step=3)
    # x_test, y_test = data_transform(testset, args.n_his, args.n_pred, device, step=3)
    
    x_train, y_train = data_transform(trainset, args.n_his, args.n_pred, step=args.n_step)
    x_val, y_val = data_transform(valset, args.n_his, args.n_pred, step=args.n_step)
    x_test, y_test = data_transform(testset, args.n_his, args.n_pred, step=args.n_step)
    
    print('check after transform', x_train.shape)
    # zscore_x = preprocessing.StandardScaler()
    # x_train = zscore_x.fit_transform(x_train)
    # x_val = zscore_x.fit_transform(x_val)
    # x_test = zscore_x.fit_transform(x_test)
    
    # zscore_y = preprocessing.StandardScaler()
    # y_train = zscore_y.fit_transform(y_train)
    # y_val = zscore_y.fit_transform(y_val)
    # y_test = zscore_y.fit_transform(y_test)
    
    print('=======================================================================================')
    print(f'check dataset count: train-{x_train.shape[0]}, val-{x_val.shape[0]}, test-{x_test.shape[0]}')
    print(f'check dataset shape: x_train-{x_train.shape}, y_train-{y_train.shape}, adj_matrix_coo-{adj_matrix_coo.shape}')
    print('=======================================================================================')
    """
    5. change dataset into dataset_iter
    """
    # train_data = create_data_list(x_train, y_train, adj_matrix_coo)
    train_data = TPLLM_Dataset(x_train, y_train, adj_matrix_coo)
    train_iter = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda x: collate_fn(x, device))
    val_data = TPLLM_Dataset(x_val, y_val, adj_matrix_coo)
    val_iter = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False,
                          collate_fn=lambda x: collate_fn(x, device))
    test_data = TPLLM_Dataset(x_test, y_test, adj_matrix_coo)
    test_iter = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False,
                           collate_fn=lambda x: collate_fn(x, device))
    
    # vertex = adj.shape[0], edge_number = nonzero_indice.shape[1]
    return n_vertex, nonzero_indices.shape[1], train_iter, val_iter, test_iter, zscore


def model_prepare(args, edge_count, device, num_layers_to_adjust=3, lora=False):
    # custom_model = CustomFeatureExtractor(args.context_dim)
    # model_llm = apply_lora(custom_model.model)
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    print('====================== check model layers ======================')
    print(model)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['c_attn']
    )
    
    if lora:
        model = get_peft_model(model, lora_config)
        for name, param in model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = False
    
    base_model = Model_TPLLM(args, edge_count=edge_count, llm_in=model)
    base_model = base_model.to(device)
    return base_model


def calculate_metrics(y_true, y_pred):
    # y_true_trffic_flow = zscore.inverse_transform(y_true.reshape(-1, ))
    
    # mae = torch.mean(torch.abs(y_true - y_pred)).item()
    # rmse = torch.sqrt(F.mse_loss(y_true, y_pred)).item()
    # mape = torch.mean(torch.abs((y_true - y_pred) / y_true)).item() * 100
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs(y_true - y_pred) / y_true)* 100
    
    return mae, rmse, mape


def train(args, model, train_iter, eval_iter, zscore):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # bar_format = '{l_bar}{bar:30}{r_bar}{bar:-30b}'
    current_time = datetime.now().strftime(r'%m%d%H%M%S')
    # os.makedirs(pth_dir_path)
    
    """
    save model state at last
    """
    pth_name = '0'
    model_state_dict = None
    minimum_ep = 0
    """
    for validate loss comparetion
    """
    minimum_val_loss = float('inf')
    for epoch in range(args.epochs):
        loss_sum, n = 0.0, 0
        model.train()
        pbar = tqdm(train_iter, ncols=80)
        for data in pbar:
            optimizer.zero_grad()
            # print('check input shape', data.x.shape)
            # print('check data shape', data.x.shape, data.y.shape, data.edge_index.shape)
            out = model(data)

            # print('check out shape', out.shape, data.y.shape)
            """
            shape of out is (batch_size * num_sensor, n_his * F_channel)
            shape of out should be the same as data.y
            shape of data.y is (batch_size * num_sensor, 3)
            """
            # print('check out and data shape', out.shape, data.y.shape)
            loss = F.l1_loss(out, data.y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n += 1
        scheduler.step()
        val_loss, minimum_val_loss, performance = val(model, eval_iter, epoch, minimum_val_loss, zscore, args)
        # gpu_usage = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        # print(f'ep:{epoch} | train loss:{loss_sum/n :.6f} | val loss:{val_loss:.6f} | GPU usage:{gpu_usage:.3f} MiB')
        print(f'ep:{epoch} | train loss:{loss_sum/n :.6f} | val loss:{val_loss:.6f} | mae:{performance[0] :.6f} | rmse:{performance[1] :.6f} | mape:{performance[2] :.6f}')
        # print(f'mae:{performance[0] :.6f} | rmse:{performance[1] :.6f} | mape:{performance[2] :.6f}')
        
        """
        find the minimum validation loss
        """
        if val_loss <= minimum_val_loss:
            minimum_val_loss = val_loss
            model_state_dict = model.state_dict()
            minimum_ep = epoch
    # print(model_state_dict)
    pth_name = f'{current_time}_ep{minimum_ep}_pred{args.n_pred}_{args.dataset}.pt'
    torch.save(model_state_dict, os.path.join('pths', pth_name))
    return


@torch.no_grad()
def val(model, eval_iter, epoch, minimum_val_loss, zscore, args):
    # bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
    model.eval()
    loss_sum, n = 0.0, 0
    # for data in tqdm(eval_iter, bar_format=bar_format):
    all_y_pred, all_y_true = [], []
    for data in eval_iter:
        out = model(data)
        loss = F.mse_loss(out, data.y)
        loss_sum += loss.item()
        n += 1
        all_y_pred.append(out)
        all_y_true.append(data.y)
    
    cur_loss = loss_sum / n
    if cur_loss < minimum_val_loss:
        minimum_val_loss = cur_loss
        print(f'minimum epoch is ep_{epoch} with val loss of:{cur_loss:.6f}')
    
    all_y_pred = torch.cat(all_y_pred, dim=0)
    all_y_true = torch.cat(all_y_true, dim=0)
    
    all_y_pred_real = zscore.inverse_transform(all_y_pred.cpu().reshape(-1, args.n_vertex)).reshape(all_y_pred.shape)
    all_y_true_real = zscore.inverse_transform(all_y_true.cpu().reshape(-1, args.n_vertex)).reshape(all_y_true.shape)
    
    # print('check prediction and true', all_y_pred_real[0,:], all_y_true_real[0,:])
    
    # print('before metrix', all_y_pred_real.shape)
    performance = calculate_metrics(all_y_pred_real, all_y_true_real)
    
    return cur_loss, minimum_val_loss, performance


@torch.no_grad()
def test(model, args, test_iter, zscore):
    """
    need to set the exact path by hand
    because datetime is used to define the dir name
    """
    model_path = './pths/TCN_GCN_GPT2_lora/0727080748_ep90_pred12_PEMS04.pt'
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    predictions = []
    labels = []
    
    for data in test_iter:
        output = model(data)
        predictions.append(output.detach().cpu().numpy())
        labels.append(data.y.detach().cpu().numpy())
        
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    all_y_pred_real = zscore.inverse_transform(predictions.reshape(-1, args.n_vertex)).reshape(predictions.shape)
    all_y_true_real = zscore.inverse_transform(labels.reshape(-1, args.n_vertex)).reshape(labels.shape)

    performance = calculate_metrics(all_y_pred_real, all_y_true_real)

    print(f'show test result mae:{performance[0] :.6f} | rmse:{performance[1] :.6f} | mape:{performance[2] :.6f}')
    # print(predictions.shape, labels.shape)
    return


def main():
    # get argument and device
    args, device = get_parameters()
    # get dataset
    n_vertex, edge_count, train_iter, eval_iter, test_iter, zscore = data_prepare(args, device)
    args.n_vertex = n_vertex
    # get model
    model = model_prepare(args, edge_count, device, lora=True)
    # print(model)
    # train model
    train(args, model, train_iter, eval_iter, zscore)
    # test model
    # para_idx = 100
    # test(model, args, test_iter=test_iter, zscore=zscore)


if __name__ == '__main__':
    main()
    torch.cuda.empty_cache()
    
    