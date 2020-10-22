import numpy as np
from numpy.linalg import norm


def compute_f_kernel(x, y, depth=2):
    x = x / norm(x, axis=1, keepdims=True)
    y = y / norm(y, axis=1, keepdims=True)
    Sigmas_dot = []
    Sigmas = []
    D = 1
    Sigmas.append(np.matmul(x, y.T))
    for i in range(depth):
        L = Sigmas[-1]
        L = np.sign(L) * (np.abs(L) - 1e-4)
        off_diagonal = 2 * D * (L * (np.pi - np.arccos(L)) +
                                np.sqrt(np.abs(1 - L ** 2))) / (2 * np.pi)
        Sigmas.append(off_diagonal)
        off_diagonal_dot = 2 * (np.pi - np.arccos(L)) / (2 * np.pi)
        Sigmas_dot.append(off_diagonal_dot)
    Sigmas_dot.append(1)
    ker = 0
    for i in range(len(Sigmas_dot)):
        ker += Sigmas[i] * np.prod(Sigmas_dot[i:])
    sx = norm(x, axis=1, keepdims=True) ** 2
    sy = norm(y, axis=1, keepdims=True) ** 2
    return sx, sy, Sigmas[-1], ker


def compute_g_kernel(x, y, sx, sy, sxy, depth=2):
    sxy = sxy / np.sqrt(np.matmul(sx, sy.T))
    Sigmas_dot = []
    Sigmas = []
    d1 = norm(x, axis=1, keepdims=True) ** 2
    d2 = norm(y, axis=1, keepdims=True) ** 2
    D = 1
    var = 2
    Sigmas.append(np.exp(-(d1 + d2.T - 2 * np.matmul(x, y.T)) / (2 * var)))
    for i in range(depth):
        L = Sigmas[-1] * sxy / D
        L = np.sign(L) * (np.abs(L) - 1e-4)
        off_diagonal = 2 * D * (L * (np.pi - np.arccos(L)) +
                                np.sqrt(np.abs(1 - L ** 2))) / (2 * np.pi)
        Sigmas.append(off_diagonal)
        off_diagonal_dot = 2 * (np.pi - np.arccos(L)) / (2 * np.pi)
        Sigmas_dot.append(off_diagonal_dot)
    Sigmas_dot.append(1)
    ker = 0
    for i in range(len(Sigmas_dot)):
        ker += Sigmas[i] * np.prod(Sigmas_dot[i:])
    return ker, Sigmas[-1]

