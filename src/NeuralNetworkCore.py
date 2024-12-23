import torch
import torch.nn as nn

class NeuralNetworkCore:
    def __init__(self, layers):
        self.layers = nn.ModuleList(layers)
        self.initial_state = None

    def initialize_network(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        self.initial_state = self.get_weights()

    # In src/NeuralNetworkCore.py
    def forward_propagation(self, input_vector):
        if isinstance(input_vector, list):
            x = torch.tensor(input_vector).float()
        elif isinstance(input_vector, torch.Tensor):
            x = input_vector.float()
        else:
            raise TypeError("input_vector must be a list or a torch.Tensor")
    
        # Ensure input shape matches first layer (batch_size, input_size)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Shape: (1, input_size)
    
        # Forward pass through layers
        for layer in self.layers:
            x = layer(x)
    
        # Ensure output matches expected size
        if x.dim() == 2 and x.size(0) == 1:
            x = x.squeeze(0)  # Shape: (output_size,)
    
        return x

    def adjust_weights(self, modulation_tensors):
        linear_layers = [layer for layer in self.layers if isinstance(layer, nn.Linear)]
        if len(modulation_tensors) != len(linear_layers):
            raise ValueError("Number of modulation tensors must match the number of Linear layers.")

        with torch.no_grad():
            for layer, modulation in zip(linear_layers, modulation_tensors):
                if modulation.shape != layer.weight.shape:
                    raise ValueError(f"Modulation shape {modulation.shape} != layer weight shape {layer.weight.shape}")
                # Update weights in-place
                layer.weight.data.add_(modulation)
                if layer.bias is not None:
                    if layer.bias.shape[0] != modulation.shape[0]:
                        raise ValueError(f"Modulation shape {modulation.shape} not compatible with bias {layer.bias.shape}")
                    # Example bias modulation
                    layer.bias.data.add_(modulation.mean(dim=1))

    def reset_weights(self):
        if self.initial_state:
            self.set_weights(self.initial_state)
        else:
            raise RuntimeError("Initial state not set. Call initialize_network() first.")

    def get_weights(self):
        return [
            {
                'weight': layer.weight.clone(),
                'bias': layer.bias.clone() if layer.bias is not None else None
            }
            for layer in self.layers if isinstance(layer, nn.Linear)
        ]

    def set_weights(self, weights):
        """
        Sets the network's weights and biases.

        Args:
            weights (list of dict): Each dict contains 'weight' and 'bias' tensors for a layer.
        """
        linear_layers = [layer for layer in self.layers if isinstance(layer, nn.Linear)]
        for layer, weight_dict in zip(linear_layers, weights):
            layer.weight.data = weight_dict['weight'].clone()
            if layer.bias is not None and weight_dict['bias'] is not None:
                layer.bias.data = weight_dict['bias'].clone()