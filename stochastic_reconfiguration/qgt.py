import torch
from functions import Markov_step, seq_modules_seed, batch_states_shuffler, single_chain, simple_epoch_single_chain, states_gen, training_full_batch, local_energies_SYK
from torch.nn.parameter import Parameter
from torch.linalg import eigh
import matplotlib.pyplot as plt 
import numpy as np
import time


def Quantum_Geometric_Tensor(batch_states, Net, layers):
    """This implementation is based on the Supplementary of Carleo, Troyer Science 2017
    Batch states is a tensor that contains all the states that are montecarlo sampled,
    batch_states.shape[0] Is the dimension of the sampling and batch_state.shape[1] is the Hilbert dimension.
    Net is the variational ansatz.
    """
    b_s = batch_states.shape[0]
    Psi_s = Net(batch_states[0, :].unsqueeze(dim = 0).type(torch.float)).squeeze()
    
    Psi_s.backward() 
    w_b = []
    for i in range(layers + 2):
      w_b.append(torch.flatten(Net[1+(2*i)].weight.grad))
      w_b.append(torch.flatten(Net[1+(2*i)].bias.grad))
    O_i = torch.cat(w_b)/Psi_s
    O_i_k = torch.tensordot(O_i, O_i, 0)
    
    for mk in range(b_s-1):
            
            Psi_s = Net(batch_states[mk + 1, :].unsqueeze(dim = 0).type(torch.float)).squeeze()
            Psi_s.backward()  
            w_b = []
            for i in range(layers + 2):
              
              w_b.append(torch.flatten(Net[1+(2*i)].weight.grad))
              w_b.append(torch.flatten(Net[1+(2*i)].bias.grad))
            w_b = torch.cat(w_b)/Psi_s
            O_i += w_b
            O_i_k += torch.tensordot(w_b, w_b, 0)
    QGT = O_i_k - torch.tensordot(O_i, O_i, 0)
    
    return QGT/b_s

def Energy_gradient(batch_states, Net, seed, layers ):
    b_s = batch_states.shape[0]
    E_loc = local_energies_SYK(Net, batch_states, seed).squeeze()
    Psi_s = Net(batch_states[0, :].unsqueeze(dim = 0).type(torch.float)).squeeze()
    
    Psi_s.backward() 
    w_b = []
    for i in range(layers + 2):
      w_b.append(torch.flatten(Net[1+(2*i)].weight.grad))
      w_b.append(torch.flatten(Net[1+(2*i)].bias.grad))
    O_i = torch.cat(w_b)/Psi_s
    E_O_i = E_loc[0]*O_i
    for mk in range(b_s-1):
            
            Psi_s = Net(batch_states[mk + 1, :].unsqueeze(dim = 0).type(torch.float)).squeeze()
            Psi_s.backward()  
            w_b = []
            for i in range(layers + 2):
              
              w_b.append(torch.flatten(Net[1+(2*i)].weight.grad))
              w_b.append(torch.flatten(Net[1+(2*i)].bias.grad))
            w_b = torch.cat(w_b)/Psi_s
            O_i += w_b
            E_O_i += E_loc[mk+1]*O_i
    E_grad = (E_O_i)/b_s - (torch.mean(E_loc)*O_i/(b_s**2))
    return E_grad
            
def Update_weights(Net, QGT, E_grad, gamma, layers, net_dim, L):
         """gamma is the scaling parameter used at every step, this function updates the network
         according to the SR Method(net is recquired to be generated by seq_modules class)"""
         #Generation of the weight gradient
         
         S_1 = torch.linalg.pinv(QGT)
         
         Delta_W = torch.tensordot(S_1, E_grad, 1)
         #Reshaping according to weight and bias storaging coordinates
         ind =    [net_dim*L, net_dim] + [net_dim*net_dim, net_dim]*layers  + [net_dim, 1]
         Delta_W = torch.split(Delta_W, ind)
         
         #Update process
         Net[1].weight = Parameter(Net[1].weight - gamma*Delta_W[0].reshape(net_dim, L ))
         Net[1].bias = Parameter(Net[1].bias - gamma*Delta_W[1])
         for i in range(layers ):
               Net[3+(2*i)].weight = Parameter(Net[3+(2*i)].weight - gamma*Delta_W[2 + 2*i].reshape(net_dim, net_dim ))
               Net[3+(2*i)].bias = Parameter(Net[3+(2*i)].bias - gamma*Delta_W[3 + 2*i]) 
         
         Net[3 + 2*layers].weight = Parameter(Net[3 + 2*layers].weight - gamma*Delta_W[2 + 2*layers]) 
         Net[3 + 2*layers].bias = Parameter(Net[3 + 2*layers].bias - gamma*Delta_W[3 + 2*layers]) 










