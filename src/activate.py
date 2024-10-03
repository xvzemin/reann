import torch 
from torch import nn

class Relu_like(nn.Module):
    def __init__(self, n_input_neuron, n_output_neuron):
        super (Relu_like,self).__init__()
        self.alpha = nn.parameter.Parameter(torch.ones(1,n_output_neuron))
        self.beta = nn.parameter.Parameter(torch.ones(1,n_output_neuron)/float(n_input_neuron))
        self.silu = nn.SiLU()                              # SiLU=x*sigmoid(x)

    def forward(self,x):
        return self.alpha*self.silu(x*self.beta)         # output=α⋅SiLU(x⋅β)

class Tanh_like(nn.Module):
    def __init__(self,n_input_neuron,n_output_neuron):
        super (Tanh_like,self).__init__()
        self.alpha=nn.parameter.Parameter(torch.ones(1,n_output_neuron)/torch.sqrt(torch.tensor([float(n_input_neuron)])))
        self.beta=nn.parameter.Parameter(torch.ones(1,n_output_neuron)/float(n_input_neuron))

    def forward(self,x):
        return self.alpha*x/torch.sqrt(1.0+torch.square(x*self.beta))
