import unittest
import torch
import torch.nn as nn
import sys
import os

# Adjust the path to import modules correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from TaskEncoder import TaskEncoder
from NeuralNetworkCore import NeuralNetworkCore
from Memory import ShortTermMemory, LongTermMemory
from DynamicWeightAdapter import DynamicWeightAdapter

class TestNeuralNetworkCore(unittest.TestCase):
    def setUp(self):
        """
        Set up all necessary components for integration testing.
        """
        # Initialize TaskEncoder
        self.task_encoder = TaskEncoder()
        
        # Define the layers for the neural network
        layers = [
            nn.Linear(512, 768),
            nn.ReLU(),
            nn.Linear(768, 256)
        ]
        
        # Initialize NeuralNetworkCore
        self.network = NeuralNetworkCore(layers)
        self.network.initialize_network()
        
        # Store the initial weights for comparison
        self.initial_weights = self.network.get_weights()

        # Initialize Memory Modules
        self.short_term_memory = ShortTermMemory()
        self.long_term_memory = LongTermMemory()
        
        # Initialize DynamicWeightAdapter
        self.weight_adapter = DynamicWeightAdapter()
        
        # Sample Input for Testing
        self.sample_task = "Implement a binary search algorithm in Python."
        self.input_vector = torch.randn(512)  # Adjusted to match the first Linear layer's input dimension (512)
        
        # Encode the task to get query embedding
        self.query_embedding = self.task_encoder.encode_task(self.sample_task).mean(dim=0)  # Shape: (hidden_size,)
        
        # Simulate storing contextual information in LongTermMemory
        self.contextual_data = torch.randn(512)  # Adjusted to match the expected embedding dimension
        self.long_term_memory.store_memory(self.contextual_data)
        
        # Retrieve key embedding from LongTermMemory
        self.key_embedding = self.long_term_memory.retrieve_memory(index=0)
    
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
        output = self.network.forward_propagation(self.input_vector)
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
        for i, (original, adjusted) in enumerate(zip(original_weights, adjusted_weights)):
            expected_weight = original['weight'] + 0.1
            actual_weight = adjusted['weight']
        
            is_close = torch.allclose(actual_weight, expected_weight)
            if not is_close:
                print(f"\nLayer {i} adjustment check:")
                print(f"Expected weight mean: {expected_weight.mean():.4f}")
                print(f"Actual weight mean: {actual_weight.mean():.4f}")
                print(f"Difference mean: {(expected_weight - actual_weight).abs().mean():.4f}")
                print(f"Expected weight: {expected_weight}")
                print(f"Actual weight: {actual_weight}")
        
            self.assertTrue(
                is_close,
                f"Layer {i} weights were not adjusted correctly. Expected mean: {expected_weight.mean():.4f}, got: {actual_weight.mean():.4f}"
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
            self.initial_weights
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
        # Store initial weights for comparison
        initial_weights = [layer.weight.clone() for layer in self.network.layers if isinstance(layer, nn.Linear)]
        
        # Generate modulation tensors
        modulation_tensors = [
            torch.ones_like(layer.weight) * 0.1 
            for layer in self.network.layers 
            if isinstance(layer, nn.Linear)
        ]
    
        # Print shapes and values before adjustment
        for i, (layer, modulation) in enumerate(zip(
            [layer for layer in self.network.layers if isinstance(layer, nn.Linear)],
            modulation_tensors
        )):
            print(f"Layer {i} weight shape: {layer.weight.shape}")
            print(f"Modulation {i} shape: {modulation.shape}")
            print(f"Initial weight mean: {layer.weight.mean():.4f}")
            print(f"Modulation mean: {modulation.mean():.4f}")
    
        # Adjust weights
        self.network.adjust_weights(modulation_tensors)
        
        # Verify adjustments with detailed error message
        for i, (layer, modulation, initial) in enumerate(zip(
            [layer for layer in self.network.layers if isinstance(layer, nn.Linear)],
            modulation_tensors,
            initial_weights
        )):
            expected_weight = initial + modulation
            actual_weight = layer.weight
            
            is_close = torch.allclose(actual_weight, expected_weight)
            if not is_close:
                print(f"\nLayer {i} adjustment check:")
                print(f"Expected mean: {expected_weight.mean():.4f}")
                print(f"Actual mean: {actual_weight.mean():.4f}")
                print(f"Difference mean: {(expected_weight - actual_weight).abs().mean():.4f}")
            
            self.assertTrue(
                is_close,
                f"Layer {i} weights were not adjusted correctly. Expected mean: {expected_weight.mean():.4f}, got: {actual_weight.mean():.4f}"
            )

if __name__ == '__main__':
    unittest.main()