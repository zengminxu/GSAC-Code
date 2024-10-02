# -*- coding:utf-8 -*-
"""
# @Author   : Chen
# @Contact  :
# @Time     : 2022/2/16
# @Project  : GCNProject
# @File     : train_gcn.py
# @Software : PyCharm
"""

from __future__ import division
from __future__ import print_function

import time
import math
import random
import torch.optim as optim
import torch.nn.functional as F
from opts import parser
from utils import *
from models import GCN

args = parser.parse_args()

data_path = './data/{}'.format(args.dataset)
train_features, train_labels, train_adj, test_features, test_labels, test_adj = load_data(data_path, args.dataset, args.k)

seed = 43
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Model and optimizer
model = GCN(in_dim=train_features.shape[1],
            hid_dim=args.hidden,
            out_dim=int(train_labels.max())+1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.9)

model.cuda()
train_features = train_features.cuda()
train_labels = train_labels.cuda()
train_adj = train_adj.cuda()
test_features = test_features.cuda()
test_labels = test_labels.cuda()
test_adj = test_adj.cuda()

train_adj = normalize_adj(train_adj)
test_adj = normalize_adj(test_adj)

n = math.ceil(args.percentage * len(train_labels))
time_total = time.time()

for epoch in range(args.epochs):
    time_start = time.time()
    model.train()
    optimizer.zero_grad()
    outputs = model(train_features, train_adj)
    loss = F.nll_loss(outputs[0:n, :], train_labels[0:n])
    loss.backward()
    optimizer.step()
    scheduler.step()
    print('Epoch: {:03d} \t Loss.train: {:.5f} \t Time: {:.5f}s' \
          .format(epoch, loss.item(), time.time()-time_start))

print('\nThe Optimization Finished! Total time elapsed: {:.5f}s'.format(time.time()-time_total))

torch.save(model.state_dict(), '{}_Latest.pkl'.format(args.dataset))

model.eval()
output = model(test_features, test_adj)
acc = compute_accuracy(output, test_labels)
print('\nTest Accuracy: {:.4f}'.format(100 * acc))