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
       

