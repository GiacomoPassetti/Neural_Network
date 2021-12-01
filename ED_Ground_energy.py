from functions import Sparse_SYK, H_SYK
from scipy.sparse.linalg import eigsh, eigs
import numpy as np
from scipy.linalg import eigh
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

L = 16
N = int(L/2)
J = 1


H1 = torch.tensor(H_SYK(L, N, J))
u, v = torch.linalg.eigh(H1)


print(u[0]/L)



