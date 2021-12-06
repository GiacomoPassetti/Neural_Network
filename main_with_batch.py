import torch
from functions import states_gen, seq_modules, H_SYK, training_batches
from calc_trans_states import double_trans, seed_matrix, dumb_syk_transitions
from torch.linalg import eigh


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

seed = 1



#region Parameters
# Physical parameters of the SYK Model
L = 10
N = int(L/2)
J = 1
# NN Parameters
net_dim = 256

layers = 4
lr = 0.01
n_epoch = 10

max_it = 200
precision = 10
momentum = 1
#endregion

# To check the actual implementation we are going to compare the energy to the ED of a full syk representation. Watch out that ED and NN will correspond to 2 different
# random realizations.

H = torch.tensor(H_SYK(L, N, J), dtype=torch.float)
u, v = eigh (H)

input_states = torch.tensor(states_gen(L, N), dtype=torch.long)
trans_states = double_trans(input_states)
syk = dumb_syk_transitions(seed_matrix(input_states, trans_states), seed, L)
trans_states = torch.transpose(trans_states, 1, 2)
Net = seq_modules(L, net_dim, layers)
optimizer = torch.optim.Adam(Net.parameters(), lr)



training_batches(n_epoch, optimizer, Net, input_states, trans_states, syk, u[0], max_it, precision)

"""
output1 = torch.squeeze(Net(input_states.type(torch.float)))
output2 = Net(torch.reshape(trans_states.type(torch.float), (trans_states.shape[0]*trans_states.shape[1], trans_states.shape[2])))
output2 = torch.reshape(output2 , (trans_states.shape[0], trans_states.shape[1]))

Energy = torch.sum(torch.mul(output1 ,torch.sum(torch.mul(syk, output2), dim = 1)))

optimizer.zero_grad()

Energy.backward()

optimizer.step()
Delta = Energy 
"""





