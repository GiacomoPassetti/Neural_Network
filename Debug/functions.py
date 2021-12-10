import numpy as np
import itertools
import math
import copy
import random
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import scipy
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.sparse import coo_matrix, csr_matrix


#region ED SYK model


def states_gen(L,N):
    which = np.array(list(itertools.combinations(range(L), N)))
    #print(which)
    grid = np.zeros((len(which), L), dtype="int8")

    # Magic
    grid[np.arange(len(which))[None].T, which] = 1
    
    return grid


def c_dag_c(L, N, i):
    
    states = states_gen(L , N) 

    num_rows, num_cols = states.shape

    c_dag_c = np.zeros((num_rows , num_rows))

    for k in range(num_rows): 
        
        if states[k][i] == 1 :
            c_dag_c[k][k] = 1
        
    return c_dag_c

def JW(L, N, i):
    
    states = states_gen(L , N) 

    num_rows, num_cols = states.shape

    JW = np.zeros((num_rows , num_rows))
    for k in range(num_rows):
        njw = 0
        for j in range(i):
            if states[k][j] == 1:
                njw =+ 1 
        JW[k][k] = (-1)**njw 
        
    return JW


def vertex(L, N, i1, i2, i3, i4, J):
    
    states = states_gen(L , N) 

    num_rows, num_cols = states.shape

    V = np.zeros((num_rows , num_rows))

    for k in range(num_rows):
        print(k)

        
        v_entry = copy.deepcopy(states[k]) 
        print(v_entry)
        print(v_entry[i2] == 1 and v_entry[i1] == 1)
        if not(v_entry[i3] == 1 and v_entry[i4] == 1): 
            continue
        njw = 0
        for j in range(i3, i4):
            if v_entry[j] == 1:
                njw =+ 1 
        v_entry[i3] = 0
        v_entry[i4] = 0
        if not(v_entry[1] == 0 and v_entry[i2] == 0): 
            continue
        for j in range(i1, i2):
            if v_entry[j] == 1:
                njw =+ 1 
        v_entry[i1] = 1
        v_entry[i2] = 1
        for j in range(num_rows):
            if (states[j] == v_entry).all():
                V[j, k] = (-1**njw)*J    
    return V  
     

def H_rd(L, N):
    dim = states_gen(L, N).shape[0]
    a = np.random.normal(loc=0.0, scale=5.0, size=(dim,dim))
    a = np.triu(a)+np.triu(a).T - np.diag(np.diagonal(a))
    return a



def H_SYK(L, N, J):
   N = int(L/2)
   M = states_gen(L, N)
   dim = M.shape[0]
   M = (M.dot(M.T))
   M = M - (N-3)*np.ones((dim, dim))
   M = np.clip(M, 0, N)
   indi = M.nonzero()
   data = np.random.normal(0, J, (indi[0].shape[0]))
   H = coo_matrix((data, indi), shape = (dim, dim))
   H = H.todense()
   
   

   H = (np.triu(H)+ np.triu(H).T - np.diag(np.diag(H)))*(4/((2*L)**(3/2)))
   
   return H

def Trivial_op(L):
   N = int(L/2)
   M = states_gen(L, N)
   dim = M.shape[0]
   M = (M.dot(M.T))
   M = M - (N-3)*np.ones((dim, dim))
   M = np.clip(M, 0, N)
   indi = M.nonzero()
   data = np.ones((indi[0].shape[0]))
   H = coo_matrix((data, indi), shape = (dim, dim))
   H = H.todense()
   
   return H

def Sparse_SYK(L , N, J):
   N = int(L/2)
   M = states_gen(L, N)
   dim = M.shape[0]
   M = (M.dot(M.T))
   M = M - (N-3)*np.ones((dim, dim))
   M = np.clip(M, 0, N)
   indi = M.nonzero()
   
   data = np.random.normal(0, J, (indi[0].shape[0]))
   H = coo_matrix((data, indi), shape = (dim, dim))
   H = H.todense()
   
   H = (np.triu(H)+ np.triu(H).T - np.diag(np.diag(H)))*(4/((2*L)**(3/2)))
   H = scipy.sparse.csc_matrix(H)
   return H



#endregion

#region Training scripts

def E_loss(output, H):
    loss = torch.tensordot(H,output,([1],[0]))
    loss = torch.tensordot(output,loss,([0],[0]))

    norm = torch.tensordot(output, output, ([0], [0]))

    loss = (1/norm)*loss
 
    return torch.squeeze(loss)

def Simple_training(n_epoch, optimizer, seq_modules, Loss, input_states, H, Eg, max_it, precision):
    Delta = Eg
    
    i = 0
    while abs(Delta) > abs(Eg)/precision:

      print("At iteration:", i)
      for epoch in range(n_epoch):  # loop over the dataset multiple times
        print("epoch", epoch)
        # (1) Initialise gradients
        optimizer.zero_grad()
        # (2) Forward pass
        outputs = seq_modules(input_states)
        loss = Loss(outputs, H)
        if epoch == n_epoch-1:
            print("Exact Eg = ", Eg)
            #print("First_entry :", v0)
            print("Temporary_energy:", loss)
            
            #print("Temporary <Psi|n> :", outputs[:]/norm ,outputs[:]/norm)

        # (3) Backward
        loss.backward()
        # (4) Compute the loss and update the weights
        optimizer.step()
        Delta = Eg - loss
      i += 1
      if i > max_it:
            print("Maximal_iterations exceeded")
            break
    print("Final Energy", loss) 

def training_batches(n_epoch, optimizer, seq_modules, input_states, trans_states, syk, Eg, max_it, precision, L):
    Delta = Eg
    
    i = 0
    while abs(Delta) > abs(Eg)/precision:

      print("At iteration:", i)
      for epoch in range(n_epoch):  # loop over the dataset multiple times
        
        # (1) Initialise gradients
        optimizer.zero_grad()
        # (2) Forward pass
        output1 = seq_modules(input_states.type(torch.float))
        output2 = seq_modules(torch.reshape(trans_states.type(torch.float), (trans_states.shape[0]*trans_states.shape[1], trans_states.shape[2])))
        output2 = torch.reshape(output2 , (trans_states.shape[0], trans_states.shape[1]))
        
        norm = torch.tensordot(output1, output1, ([0], [0]))
        
        
        
        Energy = torch.sum(torch.mul(output1.squeeze() ,torch.sum(torch.mul(syk, output2), dim = 1)))/norm
        
        if epoch == n_epoch-1:
            print("after ", epoch, "epochs :")
            print("Exact Eg = ", Eg)
            #print("First_entry :", v0)
            print("Temporary_energy:", Energy/L)
            
            #print("Temporary <Psi|n> :", outputs[:]/norm ,outputs[:]/norm)

        # (3) Backward
        Energy.backward()
        # (4) Compute the loss and update the weights
        optimizer.step()
        Delta = Eg - Energy
      i += 1
      if i > max_it:
            print("Maximal_iterations exceeded")
            break
    print("Final Energy", Energy/L) 
    return Energy/L
#endregion

#region Networks generators
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def seq_modules(input_d, netdim, layers):
                flatten = nn.Flatten()
                nets = [flatten,nn.Linear(input_d,  netdim),nn.ReLU()]
                for i in range(layers):
                    nets.append(nn.Linear(netdim, netdim))
                    nets.append(nn.ReLU())
                nets.append(nn.Linear(netdim, 1))
                
                seq_mod = nn.Sequential(*nets)
                return seq_mod

def seq_modules_sigmoid(input_d, netdim, layers):
                flatten = nn.Flatten()
                nets = [flatten,nn.Linear(input_d,  netdim),nn.ReLU()]
                for i in range(layers):
                    nets.append(nn.Linear(netdim, netdim))
                    nets.append(nn.ReLU())
                nets.append(nn.Linear(netdim, 1))
                nets.append(nn.Sigmoid)
                
                seq_mod = nn.Sequential(*nets)
                return seq_mod
#endregion

#region Transitions SYK generators
def bin_scale(L):
    bin = []
    for i in range(2*L):
        bin.append(2**(2*L-1-i))
    bin = np.array(bin)
    return bin

def bin_convertion(v1, v2, bin_scale):
    ordering_number = np.append(v1, v2).dot(bin_scale)
    return ordering_number

def H_syk_element(v1, v2, bin_scale, seed, J):
    element_seed = bin_convertion(v1, v2, bin_scale)
    np.random.seed(seed*element_seed)
    return np.random.normal(0, J)

def single_transition_gen(vec, L, N):
    inverse = torch.diag(torch.ones(L)-vec)
    annih = torch.diag(vec)
    grid = torch.tile(vec, (N**2, 1))
    annih = annih[~torch.all(annih == 0, axis=1)]
    inverse = inverse[~torch.all(inverse == 0, axis=1)]
    annih = torch.kron(torch.ones(N), annih).reshape(N**2, L)
    inverse = torch.tile(inverse, (N, 1))

    grid = grid - annih + inverse 
    return torch.unbind(grid)

def single_transition_to_stack(vec):
    L = vec.shape[0]
    N = int(L/2)

    inverse = torch.diag(torch.ones(L)-vec)
    annih = torch.diag(vec)
    grid = torch.tile(vec, (N**2, 1))
    annih = annih[~torch.all(annih == 0, axis=1)]
    inverse = inverse[~torch.all(inverse == 0, axis=1)]
    annih = torch.kron(torch.ones(N), annih).reshape(N**2, L)
    inverse = torch.tile(inverse, (N, 1))

    grid = grid - annih + inverse 
    return grid

def complete_transitions(vec, L, N):
  vecs = single_transition_gen(vec, L, N)
  out = torch.stack(tuple(map(single_transition_to_stack, vecs))).reshape((N**4, L))
  return torch.unique(out, dim = 0)

def trans_unique(trans_states):
    """
    This accepts tensors containg the allowed two particles transitions with shape [Batch, L, k] and shrinks along the k dimension eliminating the redundant states.
    returns tensor with shape [Batch, L, Batch_2nd_ord_transitions].
    """
    L = trans_states.shape[1]
    bin = torch.flipud(2**torch.arange(0, L, 1))
    trans_states = torch.tensordot(trans_states, bin, dims = ([1], [0]))



    trans_states = list(torch.split(trans_states, 1, 0))
    for i  in range(len(trans_states)):
      trans_states[i]=torch.unique(trans_states[i])
    trans_states = torch.stack(trans_states)
    out = []
    for i in range(L):
      
      out.append(torch.div(trans_states,(2**(L-1-i)), rounding_mode= 'floor'))
    trans_states = torch.stack(out, dim = 1)%2
    return trans_states
#endregion