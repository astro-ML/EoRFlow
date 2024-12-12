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
    A class representing a conditional invertible block.

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
        self.dropout_prob = self.params.get('dropout', 0.3)  # Default dropout probability is 0.3
        self.flow = self.model(n_dim, n_blocks, n_nodes, cond_dims)
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
                                nn.Dropout(self.dropout_prob),  # Add dropout here
                                #nn.LayerNorm(n_nodes),
                                nn.Linear(n_nodes, dims_out))
        
        flow = Ff.SequenceINN(n_dim)
        permute_soft = True if self.params['n_dim'] != 1 else False
        for k in range(n_blocks):
            flow.append(Fm.AllInOneBlock, cond=0, cond_shape=(cond_dims,),
                        subnet_constructor=subnet_fc, permute_soft=permute_soft)
        # Append a Sigmoid layer
        #flow.append(SigmoidModule)

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
       


# custom invertible sigmoid to be applied in reverse pass (inference)
class SigmoidModule(Fm.InvertibleModule):
    def __init__(self, dims_in, dims_c=[]):
        super().__init__(dims_in, dims_c)

    def forward(self, x, rev=False, jac=True):
        if not rev:
            # Forward pass: Apply inverse sigmoid (logit)
            y = []
            log_jacobian = []
            for xi in x:
                # Clamp xi to avoid logit of 0 or 1
                xi = torch.clamp(xi, min=1e-6, max=1 - 1e-6)
                yi = torch.log(xi / (1 - xi))
                y.append(yi)
                if jac:
                    log_det_jac = - (torch.log(xi) + torch.log(1 - xi)).sum(dim=1)
                    log_jacobian.append(log_det_jac)
            if jac:
                total_log_jacobian = sum(log_jacobian)
                return y, total_log_jacobian
            else:
                return y
        else:
            # Reverse pass: Apply sigmoid
            y = []
            log_jacobian = []
            for xi in x:
                yi = torch.sigmoid(xi)
                y.append(yi)
                if jac:
                    # Compute the log-determinant of the Jacobian
                    log_det_jac = (torch.log(yi) + torch.log(1 - yi)).sum(dim=1)
                    log_jacobian.append(log_det_jac)
            if jac:
                total_log_jacobian = sum(log_jacobian)
                return y, total_log_jacobian
            else:
                return y

    def output_dims(self, input_dims):
        return input_dims
