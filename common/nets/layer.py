import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from config import cfg

def make_linear_layers(feat_dims, relu_final=True, use_bn=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_conv1d_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv1d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_deconv_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.ConvTranspose2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

class ResBlock1D(nn.Module):
    def __init__(self, feat_dim):
        super(ResBlock1D, self).__init__()
        self.fc1 = make_linear_layers([feat_dim, feat_dim], relu_final=True, use_bn=True)
        self.fc2 = make_linear_layers([feat_dim, feat_dim], relu_final=False, use_bn=True)

    def forward(self, x):
        y = self.fc1(x)
        y = self.fc2(y)
        y = y + x
        y = F.relu(y)
        return y

class TransformerEncoder(nn.Module):
    def __init__(self, feat_dim, feat_mlp_dim, nhead, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(feat_dim)
        self.self_attn = nn.MultiheadAttention(feat_dim, nhead)
        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(nn.Linear(feat_dim, feat_mlp_dim), \
                                nn.ReLU(inplace=True), \
                                nn.Dropout(dropout), \
                                nn.Linear(feat_mlp_dim, feat_dim), \
                                nn.Dropout(dropout))

    def forward(self, x):
        y = self.norm1(x)
        y = self.self_attn(y,y,y)[0]
        y = self.dropout(y)
        x = y + x

        y = self.norm2(x)
        y = self.mlp(y)
        x = y + x
        return x

