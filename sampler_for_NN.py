
import random
import numpy as np
import itertools
import torch
from functions import seq_modules_sigmoid, states_gen, seq_modules, H_SYK, training_batches, E_loss, trans_unique
from calc_trans_states import double_trans, seed_matrix, dumb_syk_transitions
from torch.linalg import eigh
import matplotlib.pyplot as plt 
import numpy as np



seed = 1

#region Parameters
# Physical parameters of the SYK Model
L = 10
N = int(L/2)
J = 1

batch_size = 6

# NN Parameters
net_dim = 256

layers = 4
lr = 0.005
n_epoch = 10

max_it = 100
precision = 10
momentum = 0.5
#endregion



def batch_states_shuffler(batch_states, iterations):
    "It expects a tensor of shape [N_batch, L] and returns a batch of randomly generated shuffles. At every iteration  corresponds a two indices swap for every vector of the batch "
    L = batch_states.shape[1]
    
    index = torch.arange(batch_states.shape[0]).repeat(2, 1).T.flatten().unsqueeze(dim = 1)
    new_batch = batch_states.clone()
    for i in range(iterations):
        ind_flip = torch.randint(low = 0, high = L, size = (batch_states.shape[0], 2))
        new_batch[index, ind_flip.flatten().unsqueeze(dim = 1)] = new_batch[index, ind_flip.flip(dims = (1,) ).flatten().unsqueeze(dim = 1)]
    return new_batch


def shuffler_fast(batch_states):
    "It expects a tensor of shape [N_batch, L] and returns a batch of randomly generated shuffles. At every iteration  corresponds a two indices swap for every vector of the batch "
    L = batch_states.shape[1]
    
    index = torch.arange(batch_states.shape[0]).repeat(2, 1).T.flatten().unsqueeze(dim = 1)
    new_batch = batch_states.clone()
    
    ind_flip = torch.randint(low = 0, high = L, size = (batch_states.shape[0], 2))
    new_batch[index, ind_flip.flatten().unsqueeze(dim = 1)] = new_batch[index, ind_flip.flip(dims = (1,) ).flatten().unsqueeze(dim = 1)]
    return new_batch


#def Markov_step(batch, Net):




states = torch.zeros((batch_size, L), dtype=torch.long)
states[0:int(batch_size/2), 0:int(L/2)] = 1
states[int(batch_size/2):batch_size, int(L/2):L] = 1

initial_batch = batch_states_shuffler(states, iterations = 10)



#trans_states = double_trans(initial_batch)
#trans_states = trans_unique(trans_states)
#syk = dumb_syk_transitions(seed_matrix(initial_batch, trans_states), seed, L)
#trans_states = torch.transpose(trans_states, 1, 2)
Net = seq_modules(L, net_dim, layers)
optimizer = torch.optim.Adam(Net.parameters(), lr)


current_prob = Net(initial_batch.type(torch.float))
current_prob = torch.mul(current_prob, current_prob) 
proposed_batch = shuffler_fast(initial_batch)



update_prob = Net(proposed_batch.type(torch.float))
update_prob = torch.mul(update_prob, update_prob)

transition_prob = torch.clamp(update_prob / current_prob, 0, 1)

accept = torch.bernoulli(transition_prob)
accept[0,0] = 0


new_sample = accept * proposed_batch + (1 - accept) * initial_batch
new_prob = accept * update_prob + (1 - accept) * current_prob

print("OLD Batch:")
print(initial_batch)

print("PROPOSED")
print(proposed_batch)

print("ACCEPT ")
print(accept)

print("NEW SAMPLE")
print(new_sample)






    






