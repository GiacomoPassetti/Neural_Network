from functions import Sparse_SYK
from scipy.sparse.linalg import eigsh
import numpy as np

J = 1

energies = []
for L in list(np.arange(4, 22, 2)):
    N = int(L/2)
    u, v = eigsh(Sparse_SYK(L, N, J), 1, which = 'SA')
    energies.append(u[0])

np.save("Ground_state_syk_single_IT.npy", energies )
