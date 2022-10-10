import torch
from torch import nn

device = torch.device('cpu')
dtype = torch.float64
to_t = lambda array: torch.tensor(array, device=device, dtype=dtype)
from_t = lambda tensor: tensor.to("cpu").detach().numpy()


class RBF(nn.Module):
    def __init__(self, num_discrete_states, lengthscales_Init=1.0):
        super().__init__()
        self.output_scale = nn.Parameter(torch.ones((num_discrete_states),device=device, dtype=dtype)) # one for each discrete state
        self.lengthscales = nn.Parameter(lengthscales_Init*torch.ones((num_discrete_states),device=device, dtype=dtype))  # one for each discrete state
        """
        Exponentiated Quadratic kernel class.
        forward call evaluates Kernel Gram matrix at input arguments.
        The output is num_discete_states x num_tau x num_tau
        """

    def forward(self, x_grid):
        """
        classic kernel function
        """

        diffsq = (torch.div((x_grid.view(1,-1,1) - x_grid.view(1,1,-1)), self.lengthscales.view(-1,1,1)))**2

        return self.output_scale.view(-1,1,1)**2 * torch.exp(-0.5 * diffsq)
