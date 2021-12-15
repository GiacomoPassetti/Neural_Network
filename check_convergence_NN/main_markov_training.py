import torch
from functions import Markov_step,  seq_modules, batch_states_shuffler, simple_epoch, training_full_batch
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
batch_size = 20

# NN Parameters
net_dim = 512
layers = 3
lr = 0.005
n_epoch = 10
momentum = 0.5
convergence = 0.00001


#endregion





"""We define a random sparse batch of initial states, and the network"""

# Initial state
states = torch.zeros((batch_size, L), dtype=torch.float, device = device )
states[0:int(batch_size/2), 0:int(L/2)] = 1
states[int(batch_size/2):batch_size, int(L/2):L] = 1
batch_states = batch_states_shuffler(states, iterations = 10)
print(batch_states.get_device())
# Network and optimizer
Net = seq_modules(L, net_dim, layers).to(device)
optimizer = torch.optim.Adam(Net.parameters(), lr)
#optimizer = torch.optim.SGD(Net.parameters(), lr)


"""While loop that proceeds until the resulting energy saturates to a single value."""

E_old = 0
E_new = 1
while (abs(E_old - E_new) > convergence):
    E_old = E_new
    E_new = simple_epoch(n_epoch, optimizer, Net, batch_states, seed, device)
    batch_states, prob_transitions = Markov_step(batch_states, Net)
    print("Accepted transitions :", prob_transitions)
    


