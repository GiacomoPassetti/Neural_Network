import torch
from functions import  seq_modules, batch_states_shuffler, Markov_step_double_batch, seq_modules_sigmoid, simple_epoch_MARKOV, seq_modules_ReLU, shuffler_fast
from torch.linalg import eigh
import matplotlib.pyplot as plt 
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))



#region Parameters
# Physical parameters of the SYK Model
L = 10
N = int(L/2)
J = 1
seed = 1
batch_size = 6

# NN Parameters
net_dim = 516
layers = 3
lr = 0.005
n_epoch = 20
momentum = 0.5
convergence = 0.00001

markov_steps = 30


#endregion

"""We define a random sparse batch of initial states, and the network"""
#region Network and batch states generation
# Initial state
states = torch.zeros((batch_size, L), dtype=torch.long)
states[0:int(batch_size/2), 0:int(L/2)] = 1
states[int(batch_size/2):batch_size, int(L/2):L] = 1
batch_states = batch_states_shuffler(states, iterations = 10)

# Network and optimizer
Net = seq_modules(L, net_dim, layers)
optimizer = torch.optim.Adam(Net.parameters(), lr)
#endregion



energy_sampled = simple_epoch_MARKOV(n_epoch, optimizer, Net, batch_states, seed)
energy_sampled = energy_sampled.unsqueeze(dim = 1)

for i in range(markov_steps):
    proposed_batch = shuffler_fast(batch_states)
    double_batch = torch.cat((batch_states, proposed_batch), dim = 0)
    simple_epoch_MARKOV(n_epoch, optimizer, Net, double_batch, seed).unsqueeze(dim = 1)
    batch_states = Markov_step_double_batch(batch_states, proposed_batch, Net)[0]
    energy_sampled = torch.cat((energy_sampled ,simple_epoch_MARKOV(5, optimizer, Net, batch_states, seed).unsqueeze(dim = 1)), dim = 1)

