from __future__ import print_function
import argparse
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rotations_experiment.models import Hypernet
from rotations_experiment.datasets import MNIST_ROTATE, CIFAR_ROTATE


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    MSEloss = nn.MSELoss()

    for batch_idx, (data1, data2, target, _) in enumerate(train_loader):
        data1, data2, target = data1.to(device), data2.to(device), target.to(device).float()
        optimizer.zero_grad()

        output = model(data1, data2)
        output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, target.long())

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    MSEloss = nn.MSELoss()

    with torch.no_grad():
        for data1, data2, target, _ in test_loader:
            data1, data2, target = data1.to(device), data2.to(device), target.to(device).float()
            output = model(data1, data2)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target.long(), reduction='sum').item()

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq((target.long()).view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

    print(100. * correct / len(test_loader.dataset))
    return 1 - correct / len(test_loader.dataset), test_loss



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--width_g', type=int, default=20, metavar='N',
                        help='')
    parser.add_argument('--depth_g', type=int, default=3, metavar='N',
                        help='')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--experiment', type=str, default='width', metavar='T',
                        help='width/depth')
    parser.add_argument('--dataset', type=str, default='mnist', metavar='T',
                        help='dataset: cifar/mnist')
    parser.add_argument('--var', type=str, default='epoch', metavar='T',
                        help='varying param: lr/epoch')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    #torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    print (args.dataset)
    print (args.task)

    print (torch.cuda.is_available())

    if args.task == 'rotations':
        if args.dataset == 'mnist':
            nc = 1
            h = 28
            input_dim1 = h**2 * nc
            input_dim2 = h**2 * nc
            output_dim = 12
            train_set = MNIST_ROTATE(root='./data', train=True, download=True, length=None)
            test_set = MNIST_ROTATE(root='./data', train=False, download=True, length=None)

            subset_indices = random.sample(list(range(len(train_set))), 10000)
            train_set = torch.utils.data.Subset(train_set, subset_indices)

        elif args.dataset == 'cifar':
            nc = 3
            h = 32
            input_dim1 = h**2 * nc
            input_dim2 = h**2 * nc
            output_dim = 12
            train_set = CIFAR_ROTATE(root='../data', train=True, download=True, length=None)
            test_set = CIFAR_ROTATE(root='../data', train=False, download=True, length=10000)

            subset_indices = random.sample(list(range(len(train_set))), 10000)
            train_set = torch.utils.data.Subset(train_set, subset_indices)


    train_loader = torch.utils.data.DataLoader(train_set,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    rates_li = []
    losses_li = []


    for i in range(100):

        rates = []
        losses = []


        if args.var == 'epoch':

            model = Hypernet(args, input_dim1, input_dim2, depth_f=4,
                             hidden_f=200, depth_g=args.depth_g,
                             hidden_g=args.width_g, output_dim=output_dim).to(device)

            optimizer = optim.SGD(model.parameters(), lr=args.lr)

            for epoch in range(1, args.epochs + 1):
                train(args, model, device, train_loader, optimizer, epoch)
                rate, loss = test(args, model, device, test_loader)
                rates += [1-rate]
                losses += [loss]

            rates_li += [rates]
            losses_li += [losses]

            rates_runs = np.array(rates_li)
            rates_means = rates_runs.mean(axis=0)
            rates_stds = rates_runs.std(axis=0)

            print ('means and std: rates')
            print (rates_means.tolist())
            print (rates_stds.tolist())

            losses_runs = np.array(losses_li)
            losses_means = losses_runs.mean(axis=0)
            losses_stds = losses_runs.std(axis=0)

            print('means and std: loss')
            print(losses_means.tolist())
            print(losses_stds.tolist())

        elif args.var == 'lr':

            model = Hypernet(args, input_dim1, input_dim2, depth_f=4,
                             hidden_f=200, depth_g=args.depth_g, hidden_g=args.width_g,
                             output_dim=output_dim).to(device)

            for j in range(8):
                print (j)
                optimizer = optim.SGD(model.parameters(), lr=10**j * 10**(-7))

                for epoch in range(1, args.epochs + 1):
                    train(args, model, device, train_loader, optimizer, epoch)
                    rate, loss = test(args, model, device, test_loader)

                rates += [1 - rate]

            rates_li += [rates]
            losses_li += [losses]

            rates_runs = np.array(rates_li)
            rates_means = rates_runs.mean(axis=0)
            rates_stds = rates_runs.std(axis=0)

            print('means and std: rates')
            print(rates_means.tolist())
            print(rates_stds.tolist())

            file = '../results2/dataset: ' + args.dataset + ' width_g = ' + str(args.width_g) \
                   + ' depth_g = ' + str(args.depth_g)

            with open(file, 'w') as filetowrite:
                filetowrite.write(str(i) + '\n')
                filetowrite.write(str(rates_means.tolist()))
                filetowrite.write('\n')
                filetowrite.write(str(rates_stds.tolist()))


if __name__ == '__main__':
    main()