import unittest
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from PhysicsInspiredReset import PhysicsInspiredReset

class TestPhysicsInspiredReset(unittest.TestCase):
    def setUp(self):
        # Create a simple network for testing
        self.network = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
    
    def test_partial_reset(self):
        # Initialize resetter
        beta = 0.5
        resetter = PhysicsInspiredReset(self.network, beta)
        
        # Store initial weights
        initial_weights = []
        for name, param in self.network.named_parameters():
            if 'weight' in name or 'bias' in name:
                initial_weights.append({
                    'name': name,
                    'data': param.data.clone()
                })
        
        # Modify weights
        with torch.no_grad():
            for name, param in self.network.named_parameters():
                if 'weight' in name or 'bias' in name:
                    param.data.add_(0.1)
        
        # Apply reset
        resetter.reset_parameters()
        
        # Verify partial reset
        for initial, (name, param) in zip(initial_weights, self.network.named_parameters()):
            if name == initial['name']:
                diff = (param.data - initial['data']).abs().mean().item()
                self.assertGreater(diff, 0.0, "Weights should not fully reset")
                self.assertLess(diff, 0.1, "Weights should partially reset")

if __name__ == '__main__':
    unittest.main()