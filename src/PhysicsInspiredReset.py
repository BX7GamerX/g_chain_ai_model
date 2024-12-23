import torch
from NeuralNetworkCore import NeuralNetworkCore 
class PhysicsInspiredReset:
    """
    Implements a thermodynamics-inspired reset mechanism:
    Partially restores weights to their initial states after task completion.
    This avoids total re-initialization while preserving accumulated knowledge.

    The reset follows:
        new_weight = (1 - beta) * current_weight + beta * initial_weight
    This ensures a controlled 'cooling' effect, akin to simulated annealing.
    """

    def __init__(self, initial_weights, network: NeuralNetworkCore, beta=0.5):
        """
        Args:
            initial_weights (list of dict): Each dict in the list corresponds to a layer's
                {'weight': tensor, 'bias': tensor or None}.
            network (NeuralNetworkCore): The neural network instance.
            beta (float): The reset strength factor in [0, 1].            
        """
        self.initial_weights = initial_weights
        self.network = network  # Store the network instance
        self.beta = beta

    def reset_parameters(self, current_weights):
        """
        Resets current weights toward their initial values, applying the
        thermodynamic-inspired formula:
            new_state = (1 - beta) * current + beta * initial

        Args:
            current_weights (list of dict): Current weights and biases for each layer.

        Returns:
            list of dict: The updated state reflecting partial reset toward initial values.
        """
        updated_weights = []
        for current, initial in zip(current_weights, self.initial_weights):
            new_weight = (1 - self.beta) * current['weight'] + self.beta * initial['weight']
            new_bias = (1 - self.beta) * current.get('bias', torch.zeros_like(new_weight)) + self.beta * initial.get('bias', torch.zeros_like(new_weight))
            updated_weights.append({'weight': new_weight, 'bias': new_bias})
        
        self.network.set_weights(updated_weights)
        return updated_weights