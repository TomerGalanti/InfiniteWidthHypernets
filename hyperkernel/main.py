from __future__ import print_function
import argparse
from hyperkernel.kernels_image_completion import compute_f_kernel, compute_g_kernel
import numpy as np
import os
from utils import input_data
import numpy.random as rand
from numpy.linalg import inv

def main():
    # settings
    parser = argparse.ArgumentParser(description='Hyperkernel experiment')
    parser.add_argument('--unique_f', type=int, default=500, metavar='N',
                        help='number of unique samples x for the hypernetwork f')
    parser.add_argument('--g_sample_per_unique', type=int, default=20, metavar='G',
                        help='number of samples z per unique sample x')
    parser.add_argument('--test_size', type=int, default=10000, metavar='S',
                        help='test set size')
    parser.add_argument('--num_of_context_points', type=int, default=784, metavar='C',
                        help='number of context points')
    parser.add_argument('--num_of_avg_terms', type=int, default=5, metavar='A',
                        help='number of average terms')
    parser.add_argument('--eps', type=float, default=1e-9, metavar='E',
                        help='precision rate')
    parser.add_argument('--completion', action='store_true', default=True,
                        help='completion/representation')
    parser.add_argument('--depth_g', type=int, default=3, metavar='DG',
                        help='depth of g')
    parser.add_argument('--depth_f', type=int, default=3, metavar='DF',
                        help='depth of f')
    args = parser.parse_args()

    args.train = True
    train_size = args.unique_f*args.g_sample_per_unique
    args.train_size = train_size

    working_path = os.path.abspath(os.getcwd())
    if args.completion:
        new_directory = '/half_' + str(train_size) + '_' + str(args.unique_f)
    else:
        new_directory = '/implicit_' + str(train_size) + '_' + str(args.unique_f)
    full_path = working_path + new_directory
    if not os.path.isdir(full_path):
        os.mkdir(full_path)

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    train_data = mnist.train.images[0:args.unique_f, :]
    train_data = np.tile(train_data, (int(train_size / args.unique_f), 1))
    idx = np.random.permutation(args.test_size)
    test_data = mnist.test.images[idx[0:args.test_size], :]

    sdf_train = train_data.copy()
    sdf_test = test_data.copy()


    x_, y_ = np.meshgrid(range(28), range(28))
    coordinates = np.reshape(np.concatenate((x_[None, :, :], y_[None, :, :]), 0), (-1, 784))

    # _________________construct masks___________________
    masks = np.zeros((train_size + args.test_size, 784))
    for i in range(train_size + args.test_size):
        idx = rand.permutation(784)[0:args.num_of_context_points]
        if args.completion:
            masks[i, 0:392] = 1
        else:
            masks[i, idx] = 1
    train_masks = masks[0:train_size, :]
    test_masks = masks[train_size:train_size + args.test_size, :]

    context_train = train_data * train_masks
    context_test = test_data * test_masks

    if args.train:
        d_f = context_train
        for k in range(args.num_of_avg_terms):

            # _________________construct g_input___________________
            idx = rand.random_integers(0, 783, size=(train_size,))
            g_train = coordinates[:, idx]
            train_labels = np.zeros((train_size, 1))
            for i in range(train_size):
                train_labels[i] = sdf_train[i, idx[i]]
            d_g = g_train.T

            # _________________compute train kernels______________________
            _, sx, sxx, ntk_f_xx = compute_f_kernel(d_f, d_f)
            ntk_g_xx, nngp_xx = compute_g_kernel(d_g, d_g, sx, sx, sxx, depth=args.depth_g - 1)
            ntk_xx = ntk_f_xx * ntk_g_xx
            r_ntk = np.matmul(inv(ntk_xx + args.eps * np.diag(np.diag(ntk_xx))), train_labels)
            r_nngp = np.matmul(inv(nngp_xx + args.eps * np.diag(np.diag(nngp_xx))), train_labels)

            np.save(full_path + '/r_ntk_' + str(k) + '.npy', r_ntk)
            np.save(full_path + '/r_nngp_' + str(k) + '.npy', r_nngp)
            np.save(full_path + '/d_g_' + str(k) + '.npy', d_g)
        np.save(full_path + '/d_f.npy', d_f)

    d_f = np.load(full_path + '/d_f.npy')
    ntk_scores = np.zeros((784, args.test_size))
    nngp_scores = np.zeros((784, args.test_size))

    for k in range(args.num_of_avg_terms):
        print(k)
        r_ntk = np.load(full_path + '/r_ntk_' + str(k) + '.npy')
        r_nngp = np.load(full_path + '/r_nngp_' + str(k) + '.npy')
        d_g = np.load(full_path + '/d_g_' + str(k) + '.npy')

        # _________________inference____________________________
        for i in range(args.test_size):
            t_f = np.tile(context_test[i, :][None, :], (784, 1))
            t_g = coordinates.T
            sy, sx, syx, ntk_f_yx = compute_f_kernel(t_f, d_f)
            ntk_g_yx, nngp_yx = compute_g_kernel(t_g, d_g, sy, sx, syx, depth=args.depth_g - 1)
            ntk_yx = ntk_f_yx * ntk_g_yx
            ntk_scores[:, i] += np.matmul(ntk_yx, r_ntk)[:, 0]
            nngp_scores[:, i] += np.matmul(nngp_yx, r_nngp)[:, 0]

    ntk_scores = ntk_scores / args.num_of_avg_terms
    ntk_scores = np.reshape(ntk_scores, (28, 28, 10))
    nngp_scores = nngp_scores / args.num_of_avg_terms
    nngp_scores = np.reshape(nngp_scores, (28, 28, 10))
    np.save(full_path + '/ntk_scores.npy', ntk_scores)
    np.save(full_path + '/NNGP_scores.npy', nngp_scores)
    error_nngp = np.mean((nngp_scores - sdf_test.T) ** 2)
    error_ntk = np.mean((ntk_scores - sdf_test.T) ** 2)

    print('error_nngp: ', error_nngp)
    print('error_ntk: ', error_ntk)


if __name__ == '__main__':
    main()

