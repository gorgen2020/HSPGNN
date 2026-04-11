# -*- coding: utf-8 -*-
"""Core neural network building blocks for HSPGCN.

This file was lightly refactored to:
- remove hard-coded CUDA usage
- make tensor device placement follow the input tensors
- reduce surprising global state
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Conv2d, LayerNorm


class T_cheby_conv_ds(nn.Module):
    """Dynamic graph convolution with temporal mixing."""

    def __init__(self, c_in, c_out, K, Kt):
        super().__init__()
        c_in_new = K * 1
        self.conv1 = Conv2d(1, 1, kernel_size=(1, Kt), padding=(0, 1), stride=(1, 1), bias=True)
        self.conv2 = Conv2d(2, 1, kernel_size=(1, Kt), padding=(0, 1), stride=(1, 1), bias=True)
        self.conv3 = Conv2d(c_in_new, 1, kernel_size=(1, Kt), padding=(0, 1), stride=(1, 1), bias=True)
        self.conv4 = Conv2d(1, 1, kernel_size=(1, Kt), padding=(0, 1), stride=(1, 1), bias=True)
        self.K = K

    def forward(self, x, adj):
        n_sample, feat_in, n_node, length = x.shape
        device = x.device
        dtype = x.dtype

        l1 = adj
        l0 = torch.eye(n_node, device=device, dtype=dtype).repeat(n_sample, 1, 1)
        lap = torch.stack([l0, l1], dim=1).transpose(-1, -2)

        x2 = torch.einsum("bcnl,bknq->bckql", x, lap).contiguous().view(n_sample, -1, n_node, length)
        out2 = self.conv2(x2)

        time_toeplitz = np.eye(n_node, dtype=np.float32)
        for ii in range(n_node - 1):
            time_toeplitz[ii, ii + 1] = -1
        toeplitz = torch.tensor(time_toeplitz, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        toeplitz = toeplitz.repeat(lap.shape[0], 1, 1, 1)
        identity = torch.eye(n_node, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).repeat(lap.shape[0], 1, 1, 1)

        x1 = torch.einsum("bcnl,bknq->bckql", x, identity).contiguous().view(n_sample, -1, n_node, length)
        out1 = self.conv1(x1)

        x4 = torch.einsum("bcnl,bknq->bckql", x, toeplitz).contiguous().view(n_sample, -1, n_node, length)
        out4 = self.conv4(x4)
        return out2 + out1 + out4


def _build_temporal_mask(size: int = 60) -> torch.Tensor:
    mask = np.zeros((size, size), dtype=np.float32)
    for i in range(12):
        for j in range(12):
            mask[i, j] = 1
            mask[i + 12, j + 12] = 1
            mask[i + 24, j + 24] = 1
    for i in range(24):
        for j in range(24):
            mask[i + 36, j + 36] = 1
    return torch.tensor((-1e13) * (1 - mask), dtype=torch.float32)


class TATT_1(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super().__init__()
        self.conv1 = Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv2 = Conv2d(num_nodes, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(num_nodes, 1), requires_grad=True)
        nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)
        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        self.bn = BatchNorm1d(tem_size)
        self.register_buffer("temporal_bias_mask", _build_temporal_mask(tem_size), persistent=False)

    def forward(self, seq):
        c1 = seq.permute(0, 1, 3, 2)
        f1 = self.conv1(c1).squeeze()
        c2 = seq.permute(0, 2, 1, 3)
        f2 = self.conv2(c2).squeeze(1)
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)
        logits = logits.permute(0, 2, 1).contiguous()
        logits = self.bn(logits).permute(0, 2, 1).contiguous()
        return torch.softmax(logits + self.temporal_bias_mask.to(seq.device), -1)


class SATT_0(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super().__init__()
        self.conv1 = Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv2 = Conv2d(tem_size, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(tem_size, 1), requires_grad=True)
        nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)
        self.v = nn.Parameter(torch.rand(num_nodes, num_nodes), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        self.bn = BatchNorm1d(num_nodes)

    def forward(self, seq):
        c1 = seq.permute(0, 1, 2, 3)
        f1 = self.conv1(c1).squeeze()
        c2 = seq.permute(0, 3, 1, 2)
        f2 = self.conv2(c2).squeeze(1)
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)
        logits = logits.permute(0, 2, 1).contiguous()
        logits = self.bn(logits).permute(0, 2, 1).contiguous()
        return torch.softmax(logits, -1)


class PHYSICS_LAYER(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt):
        super().__init__()
        self.satt = SATT_0(c_out, num_nodes, tem_size)
        self.dynamic_gcn = T_cheby_conv_ds(c_out, 2 * c_out, K, Kt)

    def forward(self, x, supports, train_t_mask):
        s_coef = self.satt(x)
        adj_out = s_coef
        adj_out1 = adj_out * supports
        x_1 = self.dynamic_gcn(x, adj_out1)
        return F.leaky_relu(x_1), adj_out


class PHYSICS_DECODER(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt):
        super().__init__()
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.tatt = TATT_1(c_out, num_nodes, tem_size)
        self.satt = SATT_0(c_out, num_nodes, tem_size)
        self.dynamic_gcn = T_cheby_conv_ds(c_out, 2 * c_out, K, Kt)
        self.lstm = nn.LSTM(tem_size, tem_size, batch_first=True)
        self.bn = LayerNorm([1, num_nodes, tem_size])

    def forward(self, x, supports, train_t_mask):
        s_coef = self.satt(x)
        adj_out = s_coef
        adj_out1 = adj_out * supports

        x_1 = F.leaky_relu(self.dynamic_gcn(x, adj_out1))
        batch_size, channels, num_nodes, tem_size = x.shape
        hidden_shape = (1, batch_size * num_nodes, tem_size)
        h = torch.zeros(hidden_shape, device=x.device, dtype=x.dtype)
        c = torch.zeros(hidden_shape, device=x.device, dtype=x.dtype)
        kk = x_1.permute(0, 2, 1, 3).contiguous().view(batch_size * num_nodes, channels, tem_size)
        _, hidden = self.lstm(kk, (h, c))
        x_1 = hidden[0].squeeze().view(batch_size, channels, num_nodes, tem_size).contiguous()

        t_coef = self.tatt(x_1).transpose(-1, -2)
        x_1 = torch.einsum("bcnl,blq->bcnq", x_1, t_coef)
        out = self.bn(F.leaky_relu(x_1))
        return out, adj_out, t_coef, x
