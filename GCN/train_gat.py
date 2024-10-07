# -*- coding:utf-8 -*-
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
from models import GAT

args = parser.parse_args()

data_path = './data/{}'.format(args.dataset)
train_features, train_labels, train_adj, test_features, test_labels, test_adj = load_data(data_path, args.dataset, args.k)
train_edge = torch.nonzero(train_adj).T
test_edge = torch.nonzero(test_adj).T

seed = 43
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Model and optimizer
model = GAT(in_dim=train_features.shape[1],
            hid_dim=args.hidden,
            out_dim=int(train_labels.max())+1,
            dropout=args.dropout,
            alpha=0.2)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.9)

model.cuda()
train_features = train_features.cuda()
train_labels = train_labels.cuda()
train_edge = train_edge.cuda()
test_features = test_features.cuda()
test_labels = test_labels.cuda()
test_edge = test_edge.cuda()

n = math.ceil(args.percentage * len(train_labels))
time_total = time.time()

for epoch in range(args.epochs):
    time_start = time.time()
    model.train()
    optimizer.zero_grad()
    outputs = model(train_features, train_edge)
    loss = F.nll_loss(outputs[0:n, :], train_labels[0:n])
    loss.backward()
    optimizer.step()
    scheduler.step()
    print('Epoch: {:03d} \t Loss.train: {:.5f} \t Time: {:.5f}s' \
          .format(epoch, loss.item(), time.time()-time_start))

print('\nThe Optimization Finished! Total time elapsed: {:.5f}s'.format(time.time()-time_total))

torch.save(model.state_dict(), '{}_Latest.pkl'.format(args.dataset))

model.eval()
output = model(test_features, test_edge)
acc = compute_accuracy(output, test_labels)
print('\nTest Accuracy: {:.4f}'.format(100 * acc))
