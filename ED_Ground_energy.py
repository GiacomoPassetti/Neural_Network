from functions import Sparse_SYK
from scipy.sparse.linalg import eigsh
L = 6
N = 3
J = 1
u, v = eigsh(Sparse_SYK(L, N, J), 1, which = 'SA')
print(u)


