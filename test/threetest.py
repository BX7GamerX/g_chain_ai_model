import unittest
import torch
import torch.nn as nn
import sys
import os

# Adjust the path to import NeuralNetworkCore correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.NeuralNetworkCore import NeuralNetworkCore

class TestNeuralNetworkCore(unittest.TestCase):
    def setUp(self):
        """
        Set up the NeuralNetworkCore instance before each test.
        """
        layers = [
            nn.Linear(512, 768),
            nn.ReLU(),
            nn.Linear(768, 256)
        ]
        self.network = NeuralNetworkCore(layers)
        self.network.initialize_network()

    def test_initialize_network(self):
        """
        Test that the network initializes weights and biases correctly.
        """
        weights = self.network.get_weights()
        self.assertEqual(len(weights), 2, "There should be two Linear layers.")

        for state in weights:
            self.assertIn('weight', state, "Each initial state entry must have a 'weight' key.")
            self.assertIn('bias', state, "Each initial state entry must have a 'bias' key.")
            self.assertIsInstance(state['weight'], torch.Tensor, "'weight' should be a torch.Tensor.")
            self.assertIsInstance(state['bias'], torch.Tensor, "'bias' should be a torch.Tensor.")

    def test_forward_propagation(self):
        """
        Test the forward propagation of the network.
        """
        input_vector = torch.ones(512)  # Adjusted input size to match first layer
        output = self.network.forward_propagation(input_vector)
        self.assertIsNotNone(output, "Output should not be None.")
        self.assertIsInstance(output, torch.Tensor, "Output should be a torch.Tensor.")
        self.assertEqual(output.size(0), 256, "Output size should match the last layer's output features.")

    def test_adjust_weights(self):
        """
        Test adjusting the network's weights and biases.
        """
        original_weights = self.network.get_weights()
        modulation_tensors = [torch.ones_like(w['weight']) * 0.1 for w in original_weights]
        self.network.adjust_weights(modulation_tensors)
        adjusted_weights = self.network.get_weights()
        
        # Verify that weights have been adjusted correctly
        for original, adjusted in zip(original_weights, adjusted_weights):
            self.assertTrue(
                torch.allclose(adjusted['weight'], original['weight'] + 0.1),
                "Layer weights were not adjusted correctly."
            )
            if original['bias'] is not None and adjusted['bias'] is not None:
                # Since modulation is filled with 0.1, add 0.1 to each bias element
                expected_bias = original['bias'] + 0.1
                self.assertTrue(
                    torch.allclose(adjusted['bias'], expected_bias),
                    "Layer biases were not adjusted correctly."
                )

    def test_reset_weights(self):
        """
        Test that the network resets weights and biases to their initial state.
        """
        # Create modulation tensors for each Linear layer
        modulation_tensors = [
            torch.ones_like(layer.weight) * 0.1 for layer in self.network.layers if isinstance(layer, nn.Linear)
        ]
        self.network.adjust_weights(modulation_tensors)

        # Reset weights
        self.network.reset_weights()

        # Verify weights and biases are reset correctly
        for layer, initial in zip(
            [layer for layer in self.network.layers if isinstance(layer, nn.Linear)],
            self.network.initial_state
        ):
            # Check weights
            self.assertTrue(
                torch.allclose(layer.weight, initial['weight']),
                "Layer weights were not reset correctly."
            )
            # Check biases
            if layer.bias is not None:
                self.assertTrue(
                    torch.allclose(layer.bias, initial['bias']),
                    "Layer biases were not reset correctly."
                )
    def test_full_integration(self):
        """
        Test the full integration of TaskEncoder, Memory modules, DynamicWeightAdapter,
        """
        # Store initial weights before modification
        initial_weights = self.network.get_weights()
    
        # Generate modulation tensors
        modulation_tensors = [
            torch.ones_like(layer.weight) * 0.1 
            for layer in self.network.layers 
            if isinstance(layer, nn.Linear)
        ]

        # Adjust weights
        self.network.adjust_weights(modulation_tensors)
    
        # Verify weights after adjustment
        for (layer, modulation, initial) in zip(
            [layer for layer in self.network.layers if isinstance(layer, nn.Linear)],
            modulation_tensors,
            initial_weights
        ):
            expected_weight = initial['weight'] + modulation
            self.assertTrue(
                torch.allclose(layer.weight, expected_weight),
                "Layer weights were not adjusted correctly."
            )
if __name__ == '__main__':
    unittest.main()