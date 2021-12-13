import torch
from functions import states_gen
import random
import numpy as np
import itertools

L = 4
N = int(L/2)
bin = np.arange(0, L, 1)
states = torch.tensor(states_gen(L, N), dtype= torch.float)




def batch_states_shuffler(batch_states, iterations):
    "It expects a tensor of shape [N_batch, L] and returns a batch of randomly generated shuffles. At every iteration  corresponds a two indices swap for every vector of the batch "
    L = batch_states.shape[1]
    
    index = torch.arange(batch_states.shape[0]).repeat(2, 1).T.flatten().unsqueeze(dim = 1)
    new_batch = batch_states.clone()
    for i in range(iterations):
        ind_flip = torch.randint(low = 0, high = L, size = (batch_states.shape[0], 2))
        new_batch[index, ind_flip.flatten().unsqueeze(dim = 1)] = new_batch[index, ind_flip.flip(dims = (1,) ).flatten().unsqueeze(dim = 1)]
    return new_batch


batch_size = 8
L = 12
states = torch.zeros((batch_size, L))
states[0:int(batch_size/2), 0:int(L/2)] = 1
states[int(batch_size/2):batch_size, int(L/2):L] = 1
print(states)
new_state = batch_states_shuffler(states, iterations = 10)
print(new_state)







    






