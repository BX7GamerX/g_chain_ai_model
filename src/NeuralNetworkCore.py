import torch
import torch.nn as nn

class NeuralNetworkCore:
    def __init__(self, layers):
        self.layers = nn.ModuleList(layers)
        self.initial_state = None

    def initialize_network(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        self.initial_state = self.get_weights()

    def forward_propagation(self, input_vector):
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
        linear_layers = [layer for layer in self.layers if isinstance(layer, nn.Linear)]
        if len(weights) != len(linear_layers):
            raise ValueError("Number of weight dicts must match number of Linear layers.")

        for layer, wdict in zip(linear_layers, weights):
            if 'weight' not in wdict or 'bias' not in wdict:
                raise KeyError("Each dictionary must contain 'weight' and 'bias' keys.")
            if wdict['weight'].shape != layer.weight.shape:
                raise ValueError("Weight shape mismatch.")
            layer.weight.data.copy_(wdict['weight'])
            if layer.bias is not None and wdict['bias'] is not None:
                if wdict['bias'].shape != layer.bias.shape:
                    raise ValueError("Bias shape mismatch.")
                layer.bias.data.copy_(wdict['bias'])