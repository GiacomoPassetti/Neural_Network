# %%
import torch
from functions import  Markov_step, seq_modules_seed, batch_states_shuffler, single_chain, simple_epoch_single_chain, states_gen, training_full_batch, local_energies_SYK

from torch.linalg import eigh
import matplotlib.pyplot as plt 
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))



#region Parameters
# Physical parameters of the SYK Model
L = 6
N = int(L/2)
J = 1
seed = 2


# NN Parameters
net_dim = 60
layers = 1
lr = 0.005
n_epoch = 50
momentum = 0.5
learning_steps = 200
tau = 5
convergence = 0.001
batch_size = 20

#endregion

"""We define a random sparse batch of initial states, and the network"""
#region Network and batch states generation
# Initial state




# Network and optimizer

"""training_full_batch returns a Network trained with respect to the exact energy up to a convergence factor setted by --convergence--"""
Net = training_full_batch(L, N, seed, net_dim, layers, lr, n_epoch, convergence)
optimizer = torch.optim.Adam(Net.parameters(), lr)
states = torch.zeros((batch_size, L), dtype=torch.long)
states[0:int(batch_size/2), 0:int(L/2)] = 1
states[int(batch_size/2):batch_size, int(L/2):L] = 1
batch_states = batch_states_shuffler(states, iterations = 20)


"""to check the implementation of the montecarlo sampling we then generate a set of random states --batch_states-- and check that 
the average energy obtained through montecarlo sampling correponds to the exact value """
for i in range(learning_steps):
    print("At learning step :", i)
    single_chain(batch_states.detach(), 10, Net)
    
    local_energies = local_energies_SYK(Net, batch_states, seed)
    mc_energy = torch.mean(local_energies)
    print("MC Energy :", mc_energy)

# %%
