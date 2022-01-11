import torch
from rand_gen_cuda import rand_gen

offset = torch.arange(0,20, device='cuda', dtype=torch.float32).reshape(4,5)
print(offset)
print(rand_gen(offset, 1))