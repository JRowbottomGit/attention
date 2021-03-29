"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax#, GATConv
from GATConv import GATConv
from AGNNConv import AGNNConv

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits

class AGNN(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 feat_drop,
                 args):
        super(AGNN, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.agnn_layers = nn.ModuleList()
        # input projection (no residual)
        self.in_drop = nn.Dropout(feat_drop)
        self.agnn_layers.append(self.in_drop)
        self.W0 = nn.Linear(in_dim, num_hidden, bias=False)
        self.agnn_layers.append(self.W0)
        # self.agnn_layers.append(nn.ReLU())
        # hidden layers
        for l in range(num_layers):
            self.agnn_layers.append(AGNNConv(args))
        # output projection
        self.W1 = nn.Linear(num_hidden, num_classes, bias=False)
        self.agnn_layers.append(self.W1)
        self.args = args
    def forward(self, inputs):
        h = inputs
        h = self.agnn_layers[0](h) #dropout
        h = self.agnn_layers[1](h) #W0
        # h = self.agnn_layers[2](h) #ReLu
        for l in range(2, self.num_layers):
            h = self.agnn_layers[l](self.g, h).flatten(1)
        # output projection
        # h = self.agnn_layers[-1](h)
        # h = self.W1(h)
        logits = self.agnn_layers[-1](h)
        return logits