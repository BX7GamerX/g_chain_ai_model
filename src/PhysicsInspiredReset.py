import torch

class PhysicsInspiredReset:
    """
    Implements a thermodynamics-inspired reset mechanism:
    Partially restores weights to their initial states after task completion.
    This avoids total re-initialization while preserving accumulated knowledge.

    The reset follows:
        new_weight = (1 - beta) * current_weight + beta * initial_weight
    This ensures a controlled 'cooling' effect, akin to simulated annealing.
    """

    def __init__(self, initial_weights, beta=0.5):
        """
        Args:
            initial_weights (list of dict): Each dict in the list corresponds to a layer's
                {'weight': tensor, 'bias': tensor or None}.
            beta (float): The reset strength factor in [0, 1].            
        """
        self.initial_weights = initial_weights
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
        new_state = []
        for init_dict, curr_dict in zip(self.initial_weights, current_weights):
            w_new = (1 - self.beta) * curr_dict['weight'] + self.beta * init_dict['weight']
            b_new = None
            if init_dict['bias'] is not None and curr_dict['bias'] is not None:
                b_new = (1 - self.beta) * curr_dict['bias'] + self.beta * init_dict['bias']
            new_state.append({'weight': w_new, 'bias': b_new})

        return new_state