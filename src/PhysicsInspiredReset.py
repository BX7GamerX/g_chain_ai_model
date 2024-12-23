import torch
import torch.nn as nn

class PhysicsInspiredReset:
    """
    Implements a thermodynamics-inspired reset mechanism:
    Partially restores weights to their initial states after task completion.
    This avoids total re-initialization while preserving accumulated knowledge.

    The reset follows:
        new_weight = (1 - beta) * current_weight + beta * initial_weight
    This ensures a controlled 'cooling' effect, akin to simulated annealing.
    """

    def __init__(self, network: nn.Module, beta: float = 0.5):
        """Initialize with network and reset strength.
        
        Args:
            network: Neural network to reset
            beta: Reset strength (0 = no reset, 1 = full reset)
        """
        self.network = network
        self.beta = beta
        
        # Store initial weights as deep copy
        self.initial_weights = []
        with torch.no_grad():
            for name, param in network.named_parameters():
                if 'weight' in name or 'bias' in name:
                    self.initial_weights.append({
                        'name': name,
                        'data': param.data.clone().detach()
                    })

    def reset_parameters(self):
        """Reset network parameters using stored initial weights and beta value."""
        print(f"Resetting with beta = {self.beta}")
        with torch.no_grad():
            for stored, (name, param) in zip(self.initial_weights, self.network.named_parameters()):
                if name == stored['name']:
                    # Calculate new values
                    current = param.data
                    initial = stored['data']
                    new_value = (1 - self.beta) * current + self.beta * initial
                    
                    # Debug info
                    diff_before = (current - initial).abs().mean().item()
                    diff_after = (new_value - initial).abs().mean().item()
                    print(f"{name}: diff before={diff_before:.6f}, after={diff_after:.6f}")
                    
                    # Update parameter
                    param.data.copy_(new_value)
        
        return self.network.state_dict()