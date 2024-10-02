import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.sparse as sp



class GL_Layer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(GL_Layer, self).__init__()
        self.dropout = dropout
        self.W = nn.Parameter(torch.empty(size=(in_dim, out_dim)))
        nn.init.xavier_normal_(self.W.data)  
        self.a = nn.Parameter(torch.empty(size=(out_dim, 1)))    
        nn.init.xavier_normal_(self.a.data)

    def forward(self, input, edge):
        x = F.dropout(input, self.dropout, training=self.training)
        x = torch.mm(x, self.W)    
        norm = torch.sqrt(torch.sum(x[edge[0, :], :] * x[edge[0, :], :], dim=1, keepdim=True) \
                          * torch.sum(x[edge[1, :], :] * x[edge[1, :], :], dim=1, keepdim=True))
    
        h = F.relu((x[edge[0, :], :] * x[edge[1, :], :]) / norm) 
        s = torch.mm(h, self.a).squeeze()
        vec = -9e15 * torch.ones_like(s)    
        s = torch.where(s > 0, s, vec)  
        A = torch.sparse_coo_tensor(indices=edge, values=s, size=(x.shape[0], x.shape[0]))
        A = torch.sparse.softmax(A, dim=1)   
        A = A.to_dense()
        # k = 2
        # A = torch.mm(torch.matrix_power(A, k), x)
        return x, A

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, out_dim)
        nn.init.xavier_normal_(self.W.weight)

    def forward(self, input, adj):
        k = 1
        x = F.dropout(input, self.dropout, training=self.training)
        h = torch.mm(torch.matrix_power(adj, k), x)
        h = self.W(h)
        return h


# class GCN(nn.Module):
#     def __init__(self, in_dim, out_dim, dropout):
#         super(GCN, self).__init__()
#         self.dropout = dropout
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.W = nn.Parameter(torch.empty(size=(in_dim, out_dim)))
#         nn.init.xavier_uniform_(self.W.data)
#
#     def forward(self, input, adj):
#         x = F.dropout(input, self.dropout, training=self.training)
#         h = torch.mm(adj, torch.mm(x, self.W))  # 公式3-13
#         return F.elu(h)


class GSSL(nn.Module):
    def __init__(self, in_dim, hid_dim, hid0_dim, hid1_dim, out_dim, dropout):
        super(GSSL, self).__init__()
        self.dropout = dropout
        self.layer0 = GL_Layer(in_dim, hid0_dim, dropout)
        self.layer1 = GCN(in_dim, hid_dim, dropout)
#       self.layer2 = GL_Layer(2 * hid_dim, hid_dim, dropout)
#        self.layer3 = GCN(2 * hid_dim, hid_dim, dropout)
        self.Linear = nn.Linear(hid_dim, out_dim)

    def forward(self, input, edge):
        x0, A0 = self.layer0(input, edge)
        h0 = self.layer1(input, A0)
#        x1, A1 = self.layer2(h0, edge)
#        h1 = self.layer3(h0)

#        h0 = F.dropout(h0, self.dropout, training=self.training)
        output = self.Linear(h0)
        return F.log_softmax(output, dim=1), x0, A0           