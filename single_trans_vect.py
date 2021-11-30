import torch

def single_trans(states):
    """
    Calculates all single transition states for the SYK model from batched input states

    Parameters
    ----------
    states : torch.tensor
        The batched input states. Expects shape (batch, L)
    Returns
    -------
    trans_states : torch.tensor
        All states where one particle has been transferred to an empty position. 
        Output shape = (batch, L, num_trans)
    """
    batch_size = states.shape[0]
    L = states.shape[1]

    # for all elements in the batch the generator is a matrix of all positions where a one could go 
    # and the annihilator is a matrix of all positions where a one can be removed
    generator = torch.diag_embed(torch.ones_like(states) - states)
    annihilator = - torch.diag_embed(states)

    # deletion of zero rows flattens tensor along batch axes. 
    # since the number of zero rows per batch element is always the same,
    # it is possible to do a simple reshaping
    annihilator = annihilator[~torch.all(annihilator == 0, axis=1)].reshape(batch_size, -1, L)
    generator = generator[~torch.all(generator == 0, axis=1)].reshape(batch_size, -1, L)

    # to find all combinations of generator and annihilator,
    # stack the generator matrix and repeat the annihilator rowwise
    op_len = annihilator.shape[1]
    generator = generator.repeat(1,op_len,1)
    annihilator = annihilator.repeat_interleave(op_len, dim=1)

    # add all possible combinations of annihilator and generator to input state 
    # to get all single transition states
    # reshaped such that the new dimension is at the and dim=1 stays the spatial dimension
    return (states.unsqueeze(1) + generator + annihilator).transpose(1,2)

def double_trans(states):
    """
    This calculates all double transitions for the syk model by using the single_trans function twice
    Parameters
    ----------
    states : torch.tensor
        The batched input states. Expects shape (batch, L)
    Returns
    -------
    trans_states : torch.tensor
        All states where two particle have been transferred (or a single particle twice)
        Output shape = (batch, L, num_trans)
    """
    L = states.shape[1]
    batch_size = states.shape[0]

    single_trans_states = single_trans(states)
    # flatten single transition states to be able to apply the single trans function again
    # the flattening is reversed at the end to associate each state with its batch number
    single_trans_states = single_trans_states.transpose(1,2).flatten(start_dim=0, end_dim=1)
    
    double_trans_states = single_trans(single_trans_states)

    # now reshaping again to separate batch dim from transition states
    double_trans_states = double_trans_states.transpose(1,2).reshape(batch_size,-1,L).transpose(1,2)
    return double_trans_states




if __name__ == '__main__':
    states = torch.tensor([[1,1,0,0],[1,0,0,1]])
    print(states)
    print(double_trans(states))