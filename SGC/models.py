# -*- coding:utf-8 -*-
"""
# @Author   : Chen
# @Contact  :
# @Time     : 2022/2/16
# @Project  : GCNProject
# @File     : models.py
# @Software : PyCharm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.W1 = nn.Linear(in_dim, 2 * hid_dim)
        nn.init.xavier_normal_(self.W1.weight)
        self.W2 = nn.Linear(2 * hid_dim, out_dim)
        nn.init.xavier_normal_(self.W2.weight)
#       self.Linear = nn.Linear(hid_dim, out_dim)

    def forward(self, input, adj):
        x = F.dropout(input, self.dropout, training=self.training)
        h1 = torch.mm(adj,x)
        h1 = self.W1(h1)
        h1 = F.dropout(h1, self.dropout, training=self.training)
        h2 = torch.mm(adj, h1)
        h2 = self.W2(h2)
        # h2 = F.elu(h2)
        h2 = F.dropout(h2, self.dropout, training=self.training)
        # output = self.Linear(h2)
        return F.log_softmax(h2, dim=1)


class gat_layer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, alpha):
        super(gat_layer, self).__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.W = nn.Parameter(torch.empty(size=(in_dim, out_dim)))
        nn.init.xavier_normal_(self.W.data)
        self.a = nn.Parameter(torch.empty(size=(2 * out_dim, 1)))
        nn.init.xavier_normal_(self.a.data)

    def forward(self, input, edge):
        x = F.dropout(input, self.dropout, training=self.training)
        h = torch.mm(x, self.W)
        hc = torch.cat([h[edge[0,:],:], h[edge[1,:],:]], dim=1)
        adj_value = F.leaky_relu(torch.mm(hc, self.a).squeeze(), negative_slope = self.alpha)
        adj = torch.sparse_coo_tensor(indices=edge, values=adj_value, size=(x.shape[0], x.shape[0]))
        A = torch.sparse.softmax(adj, dim=1).to_dense()
        output = torch.mm(A, h)
        return F.elu(output)


class GAT(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout, alpha):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.layer0 = gat_layer(in_dim, 2 * hid_dim, dropout, alpha)
        self.layer1 = gat_layer(2 * hid_dim, hid_dim, dropout, alpha)
        self.Linear = nn.Linear(hid_dim, out_dim)

    def forward(self, input, edge):
        h0 = self.layer0(input, edge)
        h1 = self.layer1(h0, edge)
        h1 = F.dropout(h1, self.dropout, training=self.training)
        output = self.Linear(h1)
        return F.log_softmax(output, dim=1)