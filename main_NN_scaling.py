import torch
from functions import states_gen, seq_modules, H_SYK, training_batches, E_loss, trans_unique
from calc_trans_states import double_trans, seed_matrix, dumb_syk_transitions
from torch.linalg import eigh
import matplotlib.pyplot as plt 
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


def Training_with_batches(L, J, net_dim, layers, lr, n_epoch, seed, max_it, precision):

   seed = 1
   #region Parameters
   # Physical parameters of the SYK Model
   L = 12
   N = int(L/2)
   J = 1
   
   # NN Parameters
   net_dim = 256
   
   layers = 4
   lr = 0.005
   n_epoch = 10
   
   max_it = 100
   precision = 10
   momentum = 0.5
   #endregion
   
   # To check the actual implementation we are going to compare the energy to the ED of a full syk representation. Watch out that ED and NN will correspond to 2 different
   # random realizations.
   
   
   N = int(L/2)
   H = torch.tensor(H_SYK(L, N, J), dtype=torch.float)
   
   u, v = eigh (H)
   
   input_states = torch.tensor(states_gen(L, N), dtype=torch.long)
   trans_states = double_trans(input_states)
   trans_states = trans_unique(trans_states)
   syk = dumb_syk_transitions(seed_matrix(input_states, trans_states), seed, L)
   trans_states = torch.transpose(trans_states, 1, 2)
   Net = seq_modules(L, net_dim, layers)
   optimizer = torch.optim.Adam(Net.parameters(), lr)
   #optimizer = torch.optim.SGD(Net.parameters(), lr)
       
   
   training_batches(n_epoch, optimizer, Net, input_states, trans_states, syk, u[0]/L, max_it, precision, L)