import argparse

# Training settings
parser = argparse.ArgumentParser(description="PyTorch implementation of Models.")
parser.add_argument('--dataset', type=str, default='Somethingv2', choices=['JHMDB', 'HMDB51', 'UCF101', 'Somethingv2'], help='Define dataset name.')
parser.add_argument('--k', type=int, default=3, help='Number of nearest neighbors.')
parser.add_argument('--epochs', type=int, default=180, help='Number of training epochs.')
parser.add_argument('--lr', type=float, default=0.002, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 regularization on model weights.')
parser.add_argument('--hidden', type=int, default=192, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate.')
parser.add_argument('--percentage', type=float, default=0.5, help='Proportion of labeled samples.')
