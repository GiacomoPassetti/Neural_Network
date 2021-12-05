from functions import Sparse_SYK, H_SYK
from scipy.sparse.linalg import eigsh, eigs
import numpy as np
from scipy.linalg import eigh
import torch
import scipy




J = 1

a = [1]
np.save("a.npy", a)

energies = []
dev_st = []
for L in list(np.arange(4, 16, 2)):
     N = int(L/2)
     batch = []
     for i in range(20):
         print(i)
         H1 = H_SYK(L, N, J)
         H2 = scipy.sparse.csc_matrix(H1)
         u, v = eigsh(H1,1 , which = 'SA')
         

         batch.append(u[0]/L)
         print(u[0]/L)
     dev = np.std(batch)
     energy = sum(batch)/len(batch)
     print(energy)
     energies.append(energy)
     dev_st.append(dev)

np.save("energy_up_to_14_20avgs.npy", energies)
np.save("std_up_to_14_20avgs.npy", dev_st)



