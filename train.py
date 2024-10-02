from __future__ import division
from __future__ import print_function

import math
import random
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
from opts import parser
from utils import *
from model import GSSL

import torch
import torch.nn as nn

args = parser.parse_args()  

data_path = './Data/{}'.format(args.dataset)  
train_features, train_labels, train_adj, test_features, test_labels, test_adj = load_data(data_path, args.dataset,
                                                                                          args.k)
train_edge = torch.nonzero(train_adj).T 
test_edge = torch.nonzero(test_adj).T

seed = 43
random.seed(seed)
torch.manual_seed(seed)  
torch.cuda.manual_seed(seed)  



in_dim = train_features.shape[1]
print("in_dim的大小为:", in_dim)
model = GSSL(in_dim=train_features.shape[1],
             hid_dim=args.hidden,
             hid0_dim=args.hidden0,
             hid1_dim=args.hidden1,
             out_dim=int(train_labels.max()) + 1,
             dropout=args.dropout)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.9)

model.cuda()
train_features = train_features.cuda()
train_labels = train_labels.cuda()
train_edge = train_edge.cuda()
test_features = test_features.cuda()
test_labels = test_labels.cuda()
test_edge = test_edge.cuda()



#writer = SummaryWriter('runs/{}'.format(args.dataset))

args.n = math.ceil(args.percentage * len(train_labels))
time_total = time.time()

for epoch in range(args.epochs):
    loss = train(epoch, model, optimizer, train_features, train_labels, train_edge, args)


    scheduler.step()

writer.close()

print('\nThe Optimization Finished! Total time elapsed: {:.5f}s'.format(time.time() - time_total))

torch.save(model.state_dict(), '{}_Latest.pkl'.format(args.dataset))

validate(model, test_features, test_labels, test_edge)
