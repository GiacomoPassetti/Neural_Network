# %%
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from functions import H_rd, NeuralNetwork, states_gen, Sparse_SYK, H_SYK, seq_modules, Simple_training, E_loss
import numpy as np  
from scipy.linalg import eigh
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Physical parameters of the SYK Model
L = 10
N = int(L/2)
J = 1
# NN Parameters
net_dim = 256

layers = 3
lr = 1
n_epoch = 1000
max_it = 10
precision = 1000
momentum = 1

# The full basis representation
#H = torch.tensor(H_rd(L, N), dtype=torch.float)   # Full random matrix 
H = torch.tensor(H_SYK(L, N, J), dtype=torch.float)  # Hamiltonian for SYK at a fixed N

input_states = torch.tensor(states_gen(L, N), dtype=torch.float)
u, v = eigh(H)
Eg = u[0]



# Generation of a fully conected net 
Net = seq_modules(L, net_dim, layers)
optimizer = torch.optim.Adam(Net.parameters(), lr)


plt.hist(u)
# %%
# Training 
Simple_training(n_epoch, optimizer, Net, E_loss, input_states, H, Eg, max_it, precision)











# %%
