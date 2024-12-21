import torch
import torch.nn as nn

class NeuralNetworkCore:
    def __init__(self, layers):
        """
        Initializes the NeuralNetworkCore with the provided layers.

        Args:
            layers (list): A list of PyTorch layers (e.g., nn.Linear, nn.ReLU).
        """
        self.layers = nn.ModuleList(layers)
        self.initial_state = None

    
    def initialize_network(self):
        """
        Initializes the network weights using Xavier uniform initialization for Linear layers
        and zeros for biases.
        """
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        self.initial_state = self.get_weights()

    def forward_propagation(self, input_vector):
        """
        Performs forward propagation through the network.

        Args:
            input_vector (list or torch.Tensor): The input vector to the network.

        Returns:
            torch.Tensor: The output of the network after forward propagation.
        """
        if isinstance(input_vector, list):
            x = torch.tensor(input_vector, dtype=torch.float32)
        elif isinstance(input_vector, torch.Tensor):
            x = input_vector.float()
        else:
            raise TypeError("input_vector must be a list or a torch.Tensor")

        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                x = torch.relu(x)
        return x

    def adjust_weights(self, modulation_tensors):
        """
        Adjusts the weights and biases of Linear layers using the provided modulation tensors.

        Args:
            modulation_tensors (list of torch.Tensor): A list of tensors to adjust the weights and biases.
                                                     Each tensor should have the same shape as the corresponding
                                                     layer's weight. If a layer has a bias, the same modulation
                                                     tensor is applied to the bias.
        """
        linear_layers = [layer for layer in self.layers if isinstance(layer, nn.Linear)]
        
        if len(modulation_tensors) != len(linear_layers):
            raise ValueError("Number of modulation tensors must match the number of Linear layers.")

        with torch.no_grad():
            for layer, modulation in zip(linear_layers, modulation_tensors):
                if not isinstance(modulation, torch.Tensor):
                    raise TypeError("Each modulation must be a torch.Tensor")
                
                if modulation.shape != layer.weight.shape:
                    raise ValueError(f"Modulation tensor shape {modulation.shape} does not match layer weight shape {layer.weight.shape}")

                layer.weight += modulation

                if layer.bias is not None:
                    if modulation.shape[0] != layer.bias.shape[0]:
                        raise ValueError(f"Modulation tensor shape {modulation.shape} does not match layer bias shape {layer.bias.shape}")
                    layer.bias += modulation.mean(dim=1)  # Example: Aggregate modulation for bias

    def reset_weights(self):
        """
        Resets the network weights to their initial state.
        """
        if self.initial_state:
            self.set_weights(self.initial_state)
        else:
            raise RuntimeError("Initial state not set. Call initialize_network() first.")

    
    def get_weights(self):
        """
        Retrieves the current weights and biases of all Linear layers.

        Returns:
            list of dict: A list containing dictionaries with 'weight' and 'bias' tensors for each Linear layer.
        """
        return [
            {
                'weight': layer.weight.clone(),
                'bias': layer.bias.clone() if layer.bias is not None else None
            }
            for layer in self.layers if isinstance(layer, nn.Linear)
        ]

    
    def set_weights(self, weights):
        """
        Sets the weights and biases of all Linear layers to the provided values.

        Args:
            weights (list of dict): A list of dictionaries containing 'weight' and 'bias' tensors.
        """
        linear_layers = [layer for layer in self.layers if isinstance(layer, nn.Linear)]

        if len(weights) != len(linear_layers):
            raise ValueError("Number of weight dictionaries must match the number of Linear layers.")

        for layer, weight_dict in zip(linear_layers, weights):
            if 'weight' not in weight_dict:
                raise KeyError("Each weight dictionary must contain a 'weight' key.")
        
            if weight_dict['weight'].shape != layer.weight.shape:
                raise ValueError(f"Weight tensor shape {weight_dict['weight'].shape} does not match layer weight shape {layer.weight.shape}")
        
            layer.weight.data.copy_(weight_dict['weight'])
        
            if layer.bias is not None:
                if 'bias' not in weight_dict:
                    raise KeyError("Each weight dictionary must contain a 'bias' key for layers with bias.")
            
                if weight_dict['bias'].shape != layer.bias.shape:
                    raise ValueError(f"Bias tensor shape {weight_dict['bias'].shape} does not match layer bias shape {layer.bias.shape}")
            
                layer.bias.data.copy_(weight_dict['bias'])
    