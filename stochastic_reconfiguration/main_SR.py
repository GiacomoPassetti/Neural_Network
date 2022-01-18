import torch
from functions import  Markov_step, seq_modules_seed, batch_states_shuffler, single_chain, simple_epoch_single_chain, states_gen, training_full_batch, local_energies_SYK
from qgt import Quantum_Geometric_Tensor, Energy_gradient, Update_weights
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
seed = 1


# NN Parameters
net_dim = 60
layers = 1

learning_steps = 200
tau = 10

batch_size = 16
gamma = [1, 0.1, 0.001] + [0.001]*learning_steps

#endregion


 # Training that prints out the exact ground energy for this seed (Using the same parameters for the network)
training_full_batch(L, N, seed, net_dim, layers, 0.005, 100, 0.0000001)

# Initialize the network (hidden layers = layers + 2)
Net = seq_modules_seed(L, net_dim, layers, seed)


# Initialize the 
states = torch.zeros((batch_size, L), dtype=torch.long)
states[0:int(batch_size/2), 0:int(L/2)] = 1
states[int(batch_size/2):batch_size, int(L/2):L] = 1
batch_states = batch_states_shuffler(states, iterations = 20) # generate an initial batch of random states

# Implementation of a Markov chain starting from the first state along the 0-th dimension of batch_states
single_chain(batch_states.detach(), tau, Net)

for i in range(learning_steps):
    print("At learning step :", i)
    single_chain(batch_states.detach(), tau, Net)
    
    # Evaluation of Quantum Geom tens and Energy gradient followed by Network weights update 
    QGT = Quantum_Geometric_Tensor(batch_states, Net, layers)
    E_grad = Energy_gradient(batch_states, Net, seed, layers )
    Update_weights(Net, QGT, E_grad, gamma[i], layers, net_dim, L)
    
    local_energies = local_energies_SYK(Net, batch_states, seed)
    mc_energy = torch.mean(local_energies)
    print("MC Energy :", mc_energy)
