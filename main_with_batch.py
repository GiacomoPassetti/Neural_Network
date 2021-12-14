
# %%

import torch
from functions import states_gen, seq_modules, H_SYK, training_batches, E_loss, trans_unique, training_full_batch
from calc_trans_states import double_trans, seed_matrix, dumb_syk_transitions
from torch.linalg import eigh
import matplotlib.pyplot as plt 
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

seed = 1

#region Parameters
# Physical parameters of the SYK Model
L = 10
N = int(L/2)
J = 1

# NN Parameters
net_dim = 256

layers = 4
lr = 0.005
n_epoch = 10
convergence = 0.0001
max_it = 100
precision = 10
momentum = 0.5
#endregion

# To check the actual implementation we are going to compare the energy to the ED of a full syk representation. Watch out that ED and NN will correspond to 2 different
# random realizations.

training_full_batch(L, N, seed, net_dim, layers, lr, n_epoch, convergence)











# %%
