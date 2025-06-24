import logging
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import FrEIA.framework as Ff
import FrEIA.modules as Fm

class ConditionalInvertibleBlock():
    """
    Conditional Invertible Block for EoRFlow.

    Args:
        params (dict): A dictionary containing the parameters for the block.

    Attributes:
        params (dict): A dictionary containing the parameters for the block.
        flow (Ff.SequenceINN): The flow model.

    Methods:
        __init__(self, params: dict) -> None: Initializes the ConditionalInvertibleBlock object.
        model(self, n_dim: int, n_blocks: int, n_nodes: int, cond_dims: tuple) -> Ff.SequenceINN: Constructs the flow model.
        load_model(self, location: str): Loads the model from the specified location.
    """

    def __init__(self, params: dict) -> None:
        """
        Initializes the ConditionalInvertibleBlock object.

        Args:
            params (dict): A dictionary containing the parameters for the block.
        """
        self.params = params['flow']
        n_dim = self.params['n_dim']
        n_blocks = self.params['n_blocks']
        n_nodes = self.params['n_nodes']
        cond_dims = self.params['cond_dims']
        self.dropout_prob = self.params.get('dropout', 0.0)  # don't use dropout for invertible network
        self.flow = self.model(n_dim, n_blocks, n_nodes, cond_dims)
        #self.log_prob = self.log_prob(x, c)
        if self.params['load']:
            if self.load_model(self.params['model_location']):
                logging.info(f"Loaded flow from {self.params['model_location']}")
            else:
                logging.info(f"Failed to load flow from {self.params['model_location']}")
                sys.exit()

        
    def model(self, n_dim: int, n_blocks: int, n_nodes: int, cond_dims: tuple) -> Ff.SequenceINN:
        """
        Constructs the flow model.

        Args:
            n_dim (int): The dimensionality of the input.
            n_blocks (int): The number of blocks in the model.
            n_nodes (int): The number of nodes in the subnet.
            cond_dims (tuple): The dimensions of the conditional input.

        Returns:
            Ff.SequenceINN: The constructed flow model.
        """
        def subnet_fc(dims_in: int, dims_out: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(dims_in, n_nodes), 
                                nn.ReLU(),
                                nn.Linear(n_nodes, dims_out))
        
        flow = Ff.SequenceINN(n_dim)
        permute_soft = True if self.params['n_dim'] != 1 else False
        for k in range(n_blocks):
            flow.append(Fm.AllInOneBlock, cond=0, cond_shape=(cond_dims,),
                        subnet_constructor=subnet_fc, permute_soft=permute_soft)
        return flow
    
    def load_model(self, location: str) -> bool:
            """
            Loads a pre-trained model from the specified location.

            Parameters:
            - location (str): The file path of the pre-trained model.

            Returns:
            - bool: True if the model is successfully loaded, False otherwise.
            """
            try:
                self.flow.load_state_dict(torch.load(location))
                return True
            except FileNotFoundError:
                return False
    
    def log_prob(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability of x given conditioning variables c.

        Args:
            x (torch.Tensor): Samples for which to compute log probabilities. Shape: [batch_size, n_dim]
            c (torch.Tensor): Conditioning variables. Shape: [batch_size, cond_dims]

        Returns:
            torch.Tensor: Log probabilities of the samples. Shape: [batch_size]
        """
        z, jac = self.flow(x, c=[c], rev=False)
        
        log_prob = -0.5 * torch.sum(z **2, dim=1) + jac
        
        return log_prob

# custom invertible sigmoid that could be used in the flow, we apply it in preprocessing instead
class SigmoidModule(Fm.InvertibleModule):
    def __init__(self, dims_in, dims_c=[]):
        super().__init__(dims_in, dims_c)

    def forward(self, x, rev=False, jac=True):
        # x is expected to be a list of tensors, but if we assume a single tensor:
        x = x[0]  # assuming x is a list with one element
        if not rev:
            # Clamp x to avoid issues at boundaries.
            x_clamped = torch.clamp(x, min=1e-6, max=1 - 1e-6)
            # Compute logit in a vectorized manner.
            y = torch.log(x_clamped / (1 - x_clamped))
            if jac:
                log_det_jac = - (torch.log(x_clamped) + torch.log(1 - x_clamped)).sum(dim=1)
                return [y], log_det_jac
            else:
                return [y]
        else:
            y = torch.sigmoid(x)
            if jac:
                log_det_jac = (torch.log(y) + torch.log(1 - y)).sum(dim=1)
                return [y], log_det_jac
            else:
                return [y]

    def output_dims(self, input_dims):
        return input_dims