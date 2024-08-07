'''
This script is based on:
- `causal_cnn.py` from https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries
- `encoder.py` from https://github.com/zhihanyue/ts2vec

'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_binomial_mask(B, T, p=0.5):
    ''' `B` - batch size
        `T` - sequence lenght
    '''
    return torch.from_numpy(
        np.random.binomial(1, p, size=(B, T))
    ).to(torch.bool)


class Chomp1D(nn.Module):
    ''' Remove the last `t` elements of a time series
    '''
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        ''' Input: [B, C, T]
        '''
        return x[:, :, :-self.chomp_size]


class SqueezeChannels(nn.Module):
    ''' Squeeze the 3rd dimension (index 2)
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.squeeze(2)


class CausalConvolutionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        final=False,
    ):
        super().__init__()

        ### causal conv 1
        # left padding so that the convolutions are causal
        padding = (kernel_size - 1) * dilation
        conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        # remove the last padded values
        # (by default PyTorch padding applies to both side)
        chomp1 = Chomp1D(padding)
        relu1 = nn.LeakyReLU()

        ### causal conv 2
        conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        chomp2 = Chomp1D(padding)
        relu2 = nn.LeakyReLU()

        ### causal network
        self.causal = nn.Sequential(
            conv1, chomp1, relu1,
            conv2, chomp2, relu2,
        )

        ### residual connection
        self.upordownsample = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
        ) if in_channels != out_channels else None

        ### final activation function
        self.relu = nn.LeakyReLU() if final else None

    def forward(self, x):
        ''' input: [B, C, T]
        '''
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        return self.relu(out_causal + res)


class CausalCNN(nn.Module):
    ''' Causal CNN composes of a sequence of causal convolution blocks
        [B, C, T] --> [B, C_out, T]

        `in_channels`  - Size of input channels
        `channels`     - Size of hidden channels (between 2 blocks)
        `out_channels` - Size of output channels of the last block
        `depth`        - Number of causal convolution block
        `kernel_size`  - Kernel size of non-residual convolutions
    '''
    def __init__(
        self,
        in_channels,
        channels,
        out_channels,
        depth,
        kernel_size,
    ):
        super().__init__()
        # list of caucal conv blocks
        layers = []
        # initial dilation size
        dilation_size = 1

        ### first `depth` layers
        for i in range(depth):
            layers += [
                CausalConvolutionBlock(
                    in_channels=in_channels if i == 0 else channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation_size,
                    final=False,
                )
            ]
            # double the dilation size at each step
            dilation_size *= 2

        ### last layer
        layers += [
            CausalConvolutionBlock(
                in_channels=channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation_size,
                final=False,
            )
        ]

        ### all layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        ''' Input: [B, C, T]
        '''
        return self.network(x)


class CausalCNNEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        output_dim=320,
        depth=10,
        kernel_size=3,
        dropout=0.1,
        mask_mode='b',
    ):
        super().__init__()
        self.mask_mode = mask_mode

        # input projection layer along time dimension
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.feature_extractor = CausalCNN(
            in_channels=hidden_dim,
            channels=hidden_dim,
            out_channels=output_dim,
            depth=depth,
            kernel_size=kernel_size,
        )
        self.repr_dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None, mask_prob=0.5, mask_n=5, mask_l=0.1):
        ''' Input: [B, C, T]

        If binomial mask is used, `self.mask_mode` == 'b'
        - If `mask_prob` < 1.0:
            Randomly set each data point of the projection layer output
            to 0 with probability `mask_prob`
        - If `mask_prob` == 1.0
            No masking applied and the only data points that will
            set to 0 are the original missing values in input `x`
        '''
        # expected input for masking: [B, T, C]
        x = x.transpose(1, 2)
        nan_mask = ~x.isnan().any(dim=-1)
        x[~nan_mask] = 0

        ### input projection layer
        # expected input: [B, T, C]
        x = self.input_fc(x)

        ### apply mask
        # expected input for masking: [B, T, C]
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                # no mask
                mask = 'all_true'
        if mask == 'b':
            mask = generate_binomial_mask(x.size(0), x.size(1), mask_prob).to(x.device)
        # no mask
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = F
        else:
            raise Exception('Invalid mask mode')
        # include indices of missing values in original input `x`
        mask &= nan_mask
        x[~mask] = 0

        ### encoder
        # expected input: [B, C, T]
        x = x.transpose(1, 2)
        x = self.feature_extractor(x)
        x = self.repr_dropout(x)
        return x