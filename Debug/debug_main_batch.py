import torch
from functions import states_gen, seq_modules, H_SYK, training_batches, E_loss, trans_unique
from calc_trans_states import double_trans, seed_matrix, dumb_syk_transitions
from torch.linalg import eigh
import matplotlib.pyplot as plt 
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

seed = 1

#region Parameters
# Physical parameters of the SYK Model
L = 6
N = int(L/2)
J = 1

# NN Parameters
net_dim = 256

layers = 4
lr = 0.01
n_epoch = 10

#endregion

# To check the actual implementation we are going to compare the energy to the ED of a full syk representation. Watch out that ED and NN will correspond to 2 different
# random realizations.


input_states = torch.tensor(states_gen(L, N), dtype=torch.long)
trans_states = double_trans(input_states)
trans_states = trans_unique(trans_states)
syk = dumb_syk_transitions(seed_matrix(input_states, trans_states), seed, L)
trans_states = torch.transpose(trans_states, 1, 2)
Net = seq_modules(L, net_dim, layers)


output1 = Net(input_states.type(torch.float))
output2 = Net(torch.reshape(trans_states.type(torch.float), (trans_states.shape[0]*trans_states.shape[1], trans_states.shape[2])))
output2 = torch.reshape(output2 , (trans_states.shape[0], trans_states.shape[1]))
norm = torch.tensordot(output1, output1, ([0], [0]))


print(norm)










