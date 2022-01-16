# %%
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
net_dim = 150
layers = 1
lr = 0.005
n_epoch = 50
momentum = 0.5
learning_steps = 200
tau = 5
convergence = 0.001
batch_size = 20

#endregion


#local_energies = local_energies_SYK(Net, batch_states, seed)
# %% 
Net = seq_modules_seed(L, net_dim, layers, seed)
optimizer = torch.optim.Adam(Net.parameters(), lr)
states = torch.zeros((batch_size, L), dtype=torch.float)
states[0:int(batch_size/2), 0:int(L/2)] = 1
states[int(batch_size/2):batch_size, int(L/2):L] = 1
batch_states = batch_states_shuffler(states, iterations = 20)

def Quantum_Geometric_Tensor(batch_states, Net):
    b_s = batch_states.shape[0]
    Psi_s = Net(batch_states[0, :].unsqueeze(dim = 0).type(torch.float))
    Psi_s.backward() 
    w_b = []
    for i in range(layers + 2):
      w_b.append(torch.flatten(Net[1+(2*i)].weight.grad))
      w_b.append(torch.flatten(Net[1+(2*i)].bias.grad))
    w_b = torch.cat(w_b)
    QGT = torch.tensordot(w_b, w_b, 0)
    
    for mk in range(b_s-1):
            print(mk)
            Psi_s = Net(batch_states[mk + 1, :].unsqueeze(dim = 0).type(torch.float))
            Psi_s.backward()  
            w_b = []
            for i in range(layers + 2):
              
              w_b.append(torch.flatten(Net[1+(2*i)].weight.grad))
              w_b.append(torch.flatten(Net[1+(2*i)].bias.grad))
            w_b = torch.cat(w_b)
            QGT += torch.tensordot(w_b, w_b, 0)
    
    return QGT/b_s

my_QGT = Quantum_Geometric_Tensor(batch_states, Net)    
print(my_QGT.shape)
#grad = Psi.grad


# %%
