import unittest
import torch
import sys
import os

# Adjust Python path to import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from PhysicsInspiredReset import PhysicsInspiredReset
from NeuralNetworkCore import NeuralNetworkCore
import torch.nn as nn

class TestPhysicsInspiredReset(unittest.TestCase):
    def setUp(self):
        """
        Create a small NeuralNetworkCore and store its initial weights.
        Then modify the weights to simulate training before applying the reset.
        """
        # Create a simple network
        layers = [
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        ]
        self.network = NeuralNetworkCore(layers)
        self.network.initialize_network()

        # Capture the initial state
        self.initial_state = self.network.get_weights()
        
        # Simulate modifications (e.g. training updates)
        with torch.no_grad():
            for layer in self.network.layers:
                if isinstance(layer, nn.Linear):
                    layer.weight.add_(0.5)  # shift weights by +0.5
                    if layer.bias is not None:
                        layer.bias.add_(0.3)
        
        self.modified_state = self.network.get_weights()

    def test_partial_reset(self):
        """
        Test that with a beta value in (0, 1), weights and biases
        are partially reset toward their initial state.
        """
        beta = 0.5
        resetter = PhysicsInspiredReset(self.initial_state, beta=beta)
        new_state = resetter.reset_parameters(self.modified_state)
        
        # Check that each layer's weights are an interpolation of modified and initial:
        # new_weight = (1 - beta)*modified + beta*initial
        for init_dict, mod_dict, new_dict in zip(self.initial_state, self.modified_state, new_state):
            # Weights
            expected_weight = (1 - beta) * mod_dict['weight'] + beta * init_dict['weight']
            self.assertTrue(
                torch.allclose(new_dict['weight'], expected_weight, atol=1e-6),
                "Partial reset weights do not match expected interpolation."
            )
            if init_dict['bias'] is not None and mod_dict['bias'] is not None:
                expected_bias = (1 - beta) * mod_dict['bias'] + beta * init_dict['bias']
                self.assertTrue(
                    torch.allclose(new_dict['bias'], expected_bias, atol=1e-6),
                    "Partial reset biases do not match expected interpolation."
                )

    def test_no_reset(self):
        """
        Test that with beta=0, no reset occurs (new state == modified state).
        """
        beta = 0.0
        resetter = PhysicsInspiredReset(self.initial_state, beta=beta)
        new_state = resetter.reset_parameters(self.modified_state)

        for mod_dict, new_dict in zip(self.modified_state, new_state):
            self.assertTrue(
                torch.allclose(mod_dict['weight'], new_dict['weight'], atol=1e-6),
                "Weights should remain the same with beta=0."
            )
            if mod_dict['bias'] is not None or new_dict['bias'] is not None:
                self.assertTrue(
                    torch.allclose(mod_dict['bias'], new_dict['bias'], atol=1e-6),
                    "Biases should remain the same with beta=0."
                )

    def test_full_reset(self):
        """
        Test that with beta=1, full reset occurs (new state == initial state).
        """
        beta = 1.0
        resetter = PhysicsInspiredReset(self.initial_state, beta=beta)
        new_state = resetter.reset_parameters(self.modified_state)

        for init_dict, new_dict in zip(self.initial_state, new_state):
            self.assertTrue(
                torch.allclose(init_dict['weight'], new_dict['weight'], atol=1e-6),
                "Weights should fully revert to initial state with beta=1."
            )
            if init_dict['bias'] is not None or new_dict['bias'] is not None:
                self.assertTrue(
                    torch.allclose(init_dict['bias'], new_dict['bias'], atol=1e-6),
                    "Biases should fully revert to initial state with beta=1."
                )

if __name__ == '__main__':
    unittest.main()