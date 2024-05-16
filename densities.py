import torch
import math
import numpy as np
from scipy.interpolate import interp1d

def w_1(z):
    return torch.sin((2 * math.pi * z[:, 0]) / 4)


def w_2(z):
    return 3 * torch.exp(-.5 * ((z[:, 0] - 1) / .6) ** 2)


def sigma(x):
    return 1 / (1 + torch.exp(- x))


def w_3(z):
    return 3 * sigma((z[:, 0] - 1) / .3)


def pot_1(z):

    data = np.load('Pems_node_missing_effect.npz')

    node_mae = (data['node_mae'])

    node_mae = np.sort(node_mae)

    delta = node_mae - node_mae.min()

    delta = delta

    node_mae = delta
    ad = node_mae[::-1]
    dfa = np.concatenate((node_mae, ad))

    temp = np.zeros(len(dfa) + 1)
    temp[1:] = dfa
    x_data = np.linspace(0, len(dfa), len(dfa) + 1, dtype=int)
    spline_function = interp1d(x_data, temp, kind='linear')
######################################################
    node = float(node_mae.shape[0])

    norm = torch.sqrt(z ** 2)
    outer_term_1 = .5 * ((norm - node) / (0.5 * node)) ** 2


    numpy_z = z.detach().numpy()
    numpy_z = np.where((numpy_z < 0) | (numpy_z > 2 * node), 0, numpy_z)
    y_spline = spline_function(numpy_z)

    inner_term_1 = torch.Tensor([y_spline.ravel()])

    outer_term_2 = torch.log(inner_term_1 + 1e-7)

    u =  outer_term_1 - outer_term_2

    return - u





