import torch
from functions import states_gen, seq_modules, H_SYK, training_batches, E_loss, trans_unique, training_full_batch
from calc_trans_states import double_trans, seed_matrix, dumb_syk_transitions
from torch.linalg import eigh
import matplotlib.pyplot as plt 
import numpy as np
from functions import H_SYK



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))



#region Parameters
# Physical parameters of the SYK Model


# NN Parameters
net_dim = 600

layers = 3
lrs = [0.0001, 0.00001, 0.000001]
n_epoch = 30
convergence = 0.00005

iterations = 200


#endregion

# To check the actual implementation we are going to compare the energy to the ED of a full syk representation. Watch out that ED and NN will correspond to 2 different
# random realizations.

L = 4
N = int(L/2)
seed = 1



batch_NN = []
batch_ED = []

for seed in range(2, iterations):
        print("Initializing evaluation nr :", seed)
        E = training_full_batch(L, N, seed, net_dim, layers, lrs, n_epoch, convergence, device).cpu().detach().numpy()
        print(E)
        batch_NN.append(E)
        np.random.seed(seed)
        u, v = eigh(torch.tensor(H_SYK(L, N, 1), device = device))
        batch_ED.append(u[0].cpu().detach().numpy()/L)


dev_NN = np.std(batch_NN)
energy_NN = sum(batch_NN)/len(batch_NN)

dev_ED = np.std(batch_ED)
energy_ED = sum(batch_ED)/len(batch_ED)

np.save("NN_energy_L_"+str(L)+"avgs_"+str(iterations)+".npy", energy_NN)
np.save("NN_dev_L_"+str(L)+"avgs_"+str(iterations)+".npy", dev_NN)
np.save("ED_energy_L_"+str(L)+"avgs_"+str(iterations)+".npy", energy_ED)
np.save("ED_dev_L_"+str(L)+"avgs_"+str(iterations)+".npy", dev_ED)


print("Energy :", energy_NN, "Dev Std :", dev_NN)


x = [6]


fig, ax = plt.subplots(dpi = 200)
ax.errorbar(x, energy_NN, dev_NN/np.sqrt(iterations), ls = "", marker = "x", label = "NN")
ax.errorbar(x, energy_ED, dev_ED/np.sqrt(iterations), ls = "", marker = "x", label = "ED")
ax.barh(-0.081, width = 19, height=0.0005, linestyle = '--', color = 'grey')
ax.set_xlabel("N")
ax.set_ylabel(r"$E_{gs}/N$")
ax.legend()
plt.grid(True)
plt.show()

