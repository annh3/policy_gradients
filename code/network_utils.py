import torch
import torch.nn as nn
from collections import OrderedDict
import pdb

def build_mlp(
          input_size,
          output_size,
          n_layers,
          size):
    """
    Args:
        input_size: int, the dimension of inputs to be given to the network
        output_size: int, the dimension of the output
        n_layers: int, the number of hidden layers of the network
        size: int, the size of each hidden layer
    Returns:
        An instance of (a subclass of) nn.Module representing the network.

    TODO:
    Build a feed-forward network (multi-layer perceptron, or mlp) that maps
    input_size-dimensional vectors to output_size-dimensional vectors.
    It should have 'n_layers' hidden layers, each of 'size' units and followed
    by a ReLU nonlinearity. The final layer should be linear (no ReLU).

    "nn.Linear" and "nn.Sequential" may be helpful.
    """
    modules = OrderedDict()
    modules['Linear_Input'] = nn.Linear(input_size, size)
    modules['ReLU_Input'] = nn.ReLU()
    for i in range(n_layers):
        modules['Linear_'+str(i)] = nn.Linear(size, size)
        modules['ReLU_'+str(i)] = nn.ReLU()
    modules['Linear_Output'] = nn.Linear(size,output_size)
    sequential = nn.Sequential(modules)
    return sequential


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_network_grads(net):
    total_norm = 0
    for p in net.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    print("total_grad_norm: ", total_norm)

def np2torch(x, cast_double_to_float=True):
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)
    """
    x = torch.from_numpy(x).to(device)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()
    return x
