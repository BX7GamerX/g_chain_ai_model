import unittest
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from PhysicsInspiredReset import PhysicsInspiredReset
from DynamicWeightAdapter import DynamicWeightAdapter
from TaskEncoder import TaskEncoder
from Memory import ShortTermMemory, LongTermMemory

class TestIntegration(unittest.TestCase):
    def setUp(self):
        # Create a simple network for testing
        self.network = nn.Sequential(
            nn.Linear(512, 768),
            nn.ReLU(),
            nn.Linear(768, 256)
        )
        self.network.eval()
        
        # Initialize components
        self.weight_adapter = DynamicWeightAdapter()
        self.task_encoder = TaskEncoder()
        self.short_mem = ShortTermMemory()
        self.long_mem = LongTermMemory()
        self.physics_reset = PhysicsInspiredReset(self.network, beta=0.5)

    def test_physics_reset_integration(self):
        # Store initial weights
        initial_weights = [param.data.clone() for param in self.network.parameters()]
        
        # Modify weights
        with torch.no_grad():
            for param in self.network.parameters():
                param.data.add_(0.1)
        
        # Apply reset
        self.physics_reset.reset_parameters()
        
        # Verify partial reset
        for init_w, curr_w in zip(initial_weights, self.network.parameters()):
            diff = (init_w - curr_w).abs().mean().item()
            self.assertLess(diff, 0.1, "Weights should be partially reset")
            self.assertGreater(diff, 0.0, "Weights should not be identical to initial")

    def test_weight_adapter_integration(self):
        # Create sample embeddings
        query = torch.randn(768)
        key = torch.randn(768)
        
        # Get modulation tensor
        modulation = self.weight_adapter.compute_modulation_tensor(query, key)
        
        # Verify shape and properties
        self.assertEqual(modulation.shape, (768, 768))
        self.assertTrue(torch.allclose(modulation.sum(dim=1), 
                                     torch.ones(768), 
                                     atol=1e-6))

    def test_memory_integration(self):
        # Test data
        data = torch.randn(10)
        
        # Test short-term memory
        self.short_mem.update_memory(data)
        stm_state = self.short_mem.get_memory_state()
        self.assertTrue(torch.allclose(stm_state[0], data))
        
        # Test long-term memory
        self.long_mem.store_memory(data)
        ltm_state = self.long_mem.retrieve_memory(0)
        self.assertTrue(torch.allclose(ltm_state, data))

    def test_task_encoder_integration(self):
        # Test task encoding
        task = "Write a function to sort a list"
        encoded = self.task_encoder.encode_task(task)
        
        # Verify output properties
        self.assertEqual(len(encoded.shape), 3)  # (batch, seq_len, hidden_size)
        self.assertEqual(encoded.shape[0], 1)    # batch_size = 1
        self.assertEqual(encoded.shape[2], 768)  # BERT hidden size

if __name__ == '__main__':
    unittest.main()