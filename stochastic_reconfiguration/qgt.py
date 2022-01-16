import torch
from functions import Markov_step, seq_modules_seed, batch_states_shuffler, single_chain, simple_epoch_single_chain, states_gen, training_full_batch, local_energies_SYK

from torch.linalg import eigh
import matplotlib.pyplot as plt 
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))



#region Parameters
# Physical parameters of the SYK Model
L = 8
N = int(L/2)
J = 1
seed = 2


# NN Parameters
net_dim = 512
layers = 3
lr = 0.005
n_epoch = 50
momentum = 0.5
learning_steps = 200
tau = 5
convergence = 0.001
batch_size = 20

#endregion
