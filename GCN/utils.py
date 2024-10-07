import os
import numpy as np
import torch


def load_data(path="./data", dataset="HMDB51", K=1):
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
    return adj


def normalize_adj(adj):
    rowsum = torch.sum(adj, dim=1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    adj_normalized = torch.mm(d_mat_inv_sqrt, torch.mm(adj, d_mat_inv_sqrt))
    return adj_normalized


def compute_accuracy(outputs, labels):
    preds = outputs.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct_num = correct.sum()
    return correct_num / len(labels)
