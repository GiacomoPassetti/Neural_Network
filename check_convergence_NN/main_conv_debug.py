import torch
from functions import states_gen, seq_modules, H_SYK, training_batches, E_loss, trans_unique, training_full_batch
from calc_trans_states import double_trans, seed_matrix, dumb_syk_transitions
from torch.linalg import eigh
import matplotlib.pyplot as plt 
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))



#region Parameters
# Physical parameters of the SYK Model


# NN Parameters
net_dim = 512

layers = 4
lr = 0.005
n_epoch = 20
convergence = 0.00001

iterations = 20


#endregion

# To check the actual implementation we are going to compare the energy to the ED of a full syk representation. Watch out that ED and NN will correspond to 2 different
# random realizations.

L = 10
N = 5
seed = 1

E = training_full_batch(L, N, seed, net_dim, layers, lr, n_epoch, convergence, device).cpu().detach().numpy()
print(E)