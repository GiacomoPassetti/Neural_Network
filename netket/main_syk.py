import netket as nk
from netket.operator import AbstractOperator
import numpy as np
import jax.numpy as jnp
import jax
from jax import jit
from functools import partial
from netket.hilbert import Fock, Spin
from netket.vqs.mc.kernels import batch_discrete_kernel


class XOperator(AbstractOperator):
  @property
  def dtype(self):
    return float
  @property
  def is_hermitian(self):
    return True


def single_trans(states, L, N):
      
      transitions = jnp.zeros((N*N,L))
      generator = jnp.ones_like(states) - states
      ind_an = jnp.repeat(jnp.array(jnp.nonzero(states, size = N)), N)
      ind_gen = jnp.tile(jnp.array(jnp.nonzero(generator, size = N)), N)
      annihilator = transitions.at[jnp.arange(N*N), ind_an].set(-1)
      generator = transitions.at[jnp.arange(N*N), ind_gen].set(1)
      transitions = jnp.repeat(states.reshape(1, L), repeats = N*N, axis = 0)
      transitions = transitions + generator + annihilator
      

      return transitions

@jax.vmap
def single_trans_jax(states):
    return single_trans(states, L, N)

def num_of_trans(N):
  return 1+(N**2)+(((N**2)*((N-1)**2))/4)

     
#@partial(jax.vmap, in_axes = (0, None, None))
def double_trans_jax(states, L, N):
      """
        This calculates all double transitions for the syk model by using the single_trans function twice
      Parameters
      ----------
      states : jnp.array
          The batched input states. Expects shape (batch, L)
      Returns
      -------
      trans_states : jnp.array
          All states where two particle have been transferred (or a single particle twice), it already returns 
          the unique transitions
          Output shape = (batch, num_trans(N), L)sa
      """
      # first we generate the one-particle transitions


      unique_trans = jnp.zeros(( int(num_of_trans(N)), L))
      sn = single_trans(states, L, N)
      
      double_trans = single_trans_jax(sn)
      
      double_trans = double_trans.reshape(N**4, L )
      unique_trans = unique_trans.at[:].set(jnp.unique(double_trans[:], axis = 0, size = int(num_of_trans(N)) ))
      return unique_trans
      

    
#@partial(jax.vmap, in_axes = (0, 0, None, None))
def seed_matrix(states, transitions, L, N):
        """
        This takes as input the batch containing the trial states and their possible double transitions. It the matrix containing the seeds to generate the syk hamiltonian.
        """

        bin = jnp.flipud(2**jnp.arange(0, L, 1))
        
        NMax = (jnp.tensordot(jnp.ones(L),bin, axes = ([0], [0])))
        
        
        # The transitions tensor get converted to the decimal base by contracting their L dimension. States gets repeated num_trans times to allow the states sorting. 

        trans_converted = jnp.tensordot(transitions, bin, axes = ([1], [0]))

        states_conv = jnp.tensordot(states, bin, 1)
        states_conv = jnp.repeat(states_conv, repeats = int(num_of_trans(N)))


        trans_converted = jnp.stack((states_conv, trans_converted), axis = 0)


        trans_converted = jnp.sort(trans_converted, axis = 0)

        trans_converted = trans_converted.at[0, :].set(NMax*trans_converted[0, :])

        trans_converted = jnp.sum(trans_converted, axis = 0, dtype = int)

        H_syk = jnp.zeros(trans_converted.shape)
        for i in range(trans_converted.shape[0]):
  
                key = jax.random.PRNGKey(trans_converted[i]*seed)

                H_syk = H_syk.at[i].set(jax.random.normal(key)*(4/((2*L)**(3/2)))) 
            
        return H_syk
        

@partial(jax.vmap, in_axes=(None, None, 0, None), out_axes=0)
def e_loc(logpsi, pars, sigma, _extra_args):
    eta, mels = get_conns_and_mels(sigma)
    return jnp.sum(mels * jnp.exp(logpsi(pars, eta) - logpsi(pars, sigma)), axis=-1)


@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(_vstate: nk.vqs.MCState, _op: XOperator):
    return e_loc


@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, _op: XOperator):
    return vstate.samples, ()


def get_conns_and_mels(sigma):
    # create the possible transitions from the state sigma
    eta = double_trans_jax(sigma, L, N)

   
    # generate deterministically the associated entries of the syk hamiltonian
    mels = seed_matrix(sigma, eta, L, N)
    
    return eta, mels




L = 6
N = int(L/2)

seed = 3
num_trans = num_of_trans(N)
n_steps = 100
step = 5

hi = Fock(n_max=1, n_particles=N, N=L)

X_OP = XOperator(hi)

vs  = nk.vqs.MCState(nk.sampler.MetropolisLocal(hi), nk.models.RBM(alpha = 1), n_samples = 4000)
sr = nk.optimizer.SR()
gs = nk.VMC(hamiltonian = X_OP, optimizer = nk.optimizer.Adam(), variational_state=vs, preconditioner=sr)

#gs.advance(steps = 1000)
#print("energy =", gs.energy)


for i in range(100):
  gs.advance(5)
  print("energy :", gs.energy)
