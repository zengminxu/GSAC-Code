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
# train_features_hmdb51:(1530,768)
train_edge = torch.nonzero(train_adj).T  # Find out the positions of those elements that are not 0 in the adjacency matrix of the training data as edges
test_edge = torch.nonzero(test_adj).T

seed = 43
random.seed(seed)
# random.seed() 
np.random.seed(seed)
torch.manual_seed(seed)  
# Set seeds for the CPU to generate random numbers so that the result is certain.
# When you set a random seed, the following random algorithm generates numbers according to the current random seed according to a certain law.
# Random seed scope is from setup time to next setup time. To replicate the results, just set the same random seeds.
torch.cuda.manual_seed(seed)  

# Model and optimizer

# with open(r"a:\wf\Desktop\KNN_best\node_embeddings\node_embeddings.pkl", "rb") as file:
#     node_embeddings = pickle.load(file)

 # Gets the dimension of the node embedding vector
# input_dim = node_embeddings.shape[1]


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

#    if epoch % args.save_freq == 0:
#        writer.add_scalar('Train_Loss', loss, epoch)

    scheduler.step()

writer.close()

print('\nThe Optimization Finished! Total time elapsed: {:.5f}s'.format(time.time() - time_total))

torch.save(model.state_dict(), '{}_Latest.pkl'.format(args.dataset))

validate(model, test_features, test_labels, test_edge)
