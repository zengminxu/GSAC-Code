import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('my_log/GSAC')
import scipy.sparse as sp

def load_data(path="./Data", dataset="HMDB51", K=1):
    print('Loading {} dataset...'.format(dataset))

    train_features = np.load(os.path.join(path, '{}_trainset_features.npy'.format(dataset))).T
    train_labels = np.load(os.path.join(path, '{}_trainset_labels.npy'.format(dataset)))
    test_features = np.load(os.path.join(path, '{}_testset_features.npy'.format(dataset))).T
    test_labels = np.load(os.path.join(path, '{}_testset_labels.npy'.format(dataset)))

    state = np.random.get_state() 
    np.random.shuffle(train_labels)  
    np.random.set_state(state) 
    np.random.shuffle(train_features)  

    train_features = torch.FloatTensor(train_features)  
    train_labels = torch.LongTensor(train_labels)  
    train_adj = create_adj(train_features, num=K) 
    test_features = torch.FloatTensor(test_features)
    test_labels = torch.LongTensor(test_labels)
    test_adj = create_adj(test_features, num=K)

    return train_features, train_labels, train_adj, test_features, test_labels, test_adj


def create_adj(features, num=1):
    norm = torch.sum(features * features, dim=1).unsqueeze(1) \
           + torch.sum(features * features, dim=1).unsqueeze(0) \
           - 2 * torch.mm(features, features.T)  
    _, idxc = torch.topk(norm, k=num + 1, largest=False)
    idxr = torch.tensor(range(idxc.shape[0])).repeat(idxc.shape[1], 1).T
    adj = torch.zeros((features.shape[0], features.shape[0]))
    adj[idxr, idxc] = 1  
    adj = adj + adj.T * (adj.T > adj).float() - adj * (adj.T > adj).float()
#    adj = adj.numpy()
#    features = features.numpy()
#    features = normalize(features)
#    features = torch.FloatTensor(features)
#     D = normalize(adj)
#     D = torch.FloatTensor(D)
#     adj = torch.FloatTensor(adj)
#     adj = torch.mm(torch.mm(D,adj),D)
    return adj


def compute_loss(x, A, args):
    L = torch.diag(torch.sum((A + A.T) / 2, dim=1)) - (A + A.T) / 2
    loss = args.beta * (torch.trace(torch.mm(torch.mm(x.T, L), x)) \
           + args.gamma * torch.trace(torch.mm(A.T, A)))
    return loss


def compute_accuracy(outputs, labels):
    preds = outputs.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct_num = correct.sum()
    return correct_num / len(labels)


def train(epoch, model, optimizer, features, labels, edge, args):
    time_start = time.time()
    model.train()
    optimizer.zero_grad()
    outputs, x0, A0 = model(features, edge)
    loss = F.nll_loss(outputs[0:args.n,:], labels[0:args.n]) \
           + compute_loss(x0, A0, args)   
    loss.backward()
    optimizer.step()
    print('Epoch: {:03d} \t Loss.train: {:.5f} \t Time: {:.5f}s' \
          .format(epoch, loss, time.time() - time_start))
    writer.add_scalar('Train_Loss', loss, epoch)
    return loss.item()


def validate(model, features, labels, edge):
    model.eval()
    outputs, _, _ = model(features, edge)
    acc = compute_accuracy(outputs, labels)
    print('\nTest Accuracy: {:.4f}'.format(100*acc))
    return acc

# def normalize(D):
#     """Row-normalize sparse matrix"""
#     rowsum = np.array(D.sum(1))  
#     r_inv = np.power(rowsum, -0.5).flatten()  
#     r_inv[np.isinf(r_inv)] = 0.  
#     r_mat_inv = sp.diags(r_inv)   
#     D = r_mat_inv.A
#     return D
