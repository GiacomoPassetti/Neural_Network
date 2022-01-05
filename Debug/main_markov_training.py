# %%
import torch
from functions import  seq_modules_seed, batch_states_shuffler, Markov_step_double_batch, simple_epoch_MARKOV,  shuffler_fast, local_energies_SYK
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
batch_size = 12

# NN Parameters
net_dim = 256
layers = 3
lr = 0.005
n_epoch = 20
momentum = 0.5
markov_steps = 1000
bs = 10


#endregion

"""We define a random sparse batch of initial states, and the network"""
#region Network and batch states generation
# Initial state
states = torch.zeros((batch_size, L), dtype=torch.long)
states[0:int(batch_size/2), 0:int(L/2)] = 1
states[int(batch_size/2):batch_size, int(L/2):L] = 1
batch_states = batch_states_shuffler(states, iterations = 10)

# Network and optimizer
Net = seq_modules_seed(L, net_dim, layers, seed)
optimizer = torch.optim.Adam(Net.parameters(), lr)
#endregion





energy_sampled = simple_epoch_MARKOV(n_epoch, optimizer, Net, batch_states, seed)
energy_sampled = energy_sampled.unsqueeze(dim = 1)

for i in range(markov_steps):
    proposed_batch = shuffler_fast(batch_states)
    double_batch = torch.cat((batch_states, proposed_batch), dim = 0)
    simple_epoch_MARKOV(n_epoch, optimizer, Net, double_batch, seed).unsqueeze(dim = 1)
    batch_states = Markov_step_double_batch(batch_states, proposed_batch, Net)[0]
    energy_sampled = torch.cat((energy_sampled ,local_energies_SYK(Net, batch_states, seed).unsqueeze(dim = 1)), dim = 1)
    print("Markov step ", i, "done")


# %%
prova = energy_sampled[:, 0:(20+(2*20))]
print(prova.shape)
engs = []
for top in range(20):
    ydata = energy_sampled[:, 0:(50*top + 50)]
    ydata = ydata[:, ::bs]
    engs.append((torch.sum(ydata)/(ydata.shape[0]*ydata.shape[1])).detach().numpy())

xdata = np.arange(50, 1050, 50)

plt.figure(dpi = 400)
plt.plot(xdata, engs, ls = "", marker = 'x')
plt.grid(True)
plt.xlabel("Markov steps")
plt.ylabel(r"$<H>$")
plt.show()
#plt.show()
# %%
