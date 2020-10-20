from __future__ import print_function
import argparse
import torch
import numpy as np
from rotations_experiment.datasets import MNIST_ROTATE, CIFAR_ROTATE


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='cifar', metavar='T',
                        help='dataset: cifar/mnist')
    parser.add_argument('--batch_size', type=str, default='', metavar='T',
                        help='dataset: cifar/mnist')
    parser.add_argument('--task', type=str, default='rotations', metavar='T',
                        help='task: rotations/pixels')
    args = parser.parse_args()


    if args.dataset == 'mnist':
        train_set = MNIST_ROTATE(root='../data', train=True, download=True, length=None)
        test_set = MNIST_ROTATE(root='../data', train=False, download=True, length=None)

        batch_size = len(train_set)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    elif args.dataset == 'cifar':
        train_set = CIFAR_ROTATE(root='../data', train=True, download=True, length=None)
        test_set = CIFAR_ROTATE(root='../data', train=False, download=True, length=None)

        batch_size = len(test_set)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    (train_data1, train_data2, train_target, angles) = next(iter(train_loader))
    (test_data1, test_data2, test_target, test_angles) = next(iter(test_loader))


    train_set_A = np.array(train_data1)
    train_set_B = np.array(train_data2)
    train_labels = np.array(train_target)
    train_angles = np.array(train_target)

    test_set_A = np.array(test_data1)
    test_set_B = np.array(test_data2)
    test_labels = np.array(test_target)
    test_angles = np.array(test_target)

    np.save('./dataset/'+args.dataset+'/train_A', train_set_A)
    np.save('./dataset/'+args.dataset+'/train_B', train_set_B)
    np.save('./dataset/'+args.dataset+'/train_labels', train_labels)
    np.save('./dataset/' + args.dataset + '/train_angles', train_angles)

    np.save('./dataset/'+args.dataset+'/test_A', test_set_A)
    np.save('./dataset/'+args.dataset+'/test_B', test_set_B)
    np.save('./dataset/' + args.dataset + '/test_labels', test_labels)
    np.save('./dataset/'+args.dataset+'/test_angles', test_angles)

if __name__ == '__main__':
    main()