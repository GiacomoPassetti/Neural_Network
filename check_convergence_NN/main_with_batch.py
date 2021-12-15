import torch
from functions import states_gen, seq_modules, H_SYK, training_batches, E_loss, trans_unique, training_full_batch
from calc_trans_states import double_trans, seed_matrix, dumb_syk_transitions
from torch.linalg import eigh
import matplotlib.pyplot as plt 
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))



#region Parameters
# Physical parameters of the SYK Model


# NN Parameters
net_dim = 512

layers = 4
lr = 0.005
n_epoch = 20
convergence = 0.00001

iterations = 20


#endregion

# To check the actual implementation we are going to compare the energy to the ED of a full syk representation. Watch out that ED and NN will correspond to 2 different
# random realizations.

energies = []
dev_st = []
for L in list(np.arange(4, 12, 2)):
    N = int(L/2)
    batch = []
    for seed in range(iterations):
        E = training_full_batch(L, N, seed, net_dim, layers, lr, n_epoch, convergence).detach().numpy()
        print(E)
        batch.append(E)
    print(batch[0])
    dev = np.std(batch)
    energy = sum(batch)/len(batch)
    energies.append(energy)
    dev_st.append(dev)
    print("Energy for L ", L, "is : ", energy)


np.save("NN_energy_up_to_"+str(10)+"_"+str(iterations)+"avgs.npy", energies)
np.save("NN_std_up_to_"+str(10)+"_"+str(iterations)+"avgs.npy", dev_st)

    













# %%
