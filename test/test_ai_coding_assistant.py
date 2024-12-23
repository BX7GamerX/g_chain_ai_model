import unittest
import sys
import os
import torch
# Adjust Python path to import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from AICodingAssistant import AICodingAssistant
from NeuralNetworkCore import NeuralNetworkCore
from Memory import ShortTermMemory, LongTermMemory
from DynamicWeightAdapter import DynamicWeightAdapter
from TaskEncoder import TaskEncoder
from FeedbackEngine import FeedbackEngine
from OutputDecoder import OutputDecoder
from PhysicsInspiredReset import PhysicsInspiredReset
import torch.nn as nn

class TestAICodingAssistant(unittest.TestCase):
    def setUp(self):
        # Create a minimal neural network with input size matching BERT's hidden size
        layers = [
            nn.Linear(768, 32),  # Changed from 16 to 768
            nn.ReLU(),
            nn.Linear(32, 8)
        ]
        self.network = NeuralNetworkCore(layers)
        self.network.initialize_network()

        # Initialize memory modules
        self.short_mem = ShortTermMemory()
        self.long_mem = LongTermMemory()

        # Initialize other components...
        self.weight_adapter = DynamicWeightAdapter()
        self.task_encoder = TaskEncoder()
        self.output_decoder = OutputDecoder(idx_to_token={0: "print", 1: "(", 2: ")", 3: "'hello_world'"})
        self.feedback_engine = FeedbackEngine(
            network=self.network,
            short_mem=self.short_mem,
            long_mem=self.long_mem,
            weight_adapter=self.weight_adapter,
            task_encoder=self.task_encoder
        )

        # Initial weights for optional reset
        initial_weights = self.network.get_weights()

        # Create the AI Coding Assistant
        self.assistant = AICodingAssistant(
            network=self.network,
            short_mem=self.short_mem,
            long_mem=self.long_mem,
            weight_adapter=self.weight_adapter,
            task_encoder=self.task_encoder,
            feedback_engine=self.feedback_engine,
            output_decoder=self.output_decoder,
            beta=0.5
        )

    def test_process_user_request_success(self):
        # Provide a simple user prompt
        user_prompt = "Just print hello world."
        code_snippet = self.assistant.process_user_request(user_prompt, apply_reset=False)
        self.assertIsInstance(code_snippet, str, "Should return a string code snippet.")

    def test_process_user_request_failure_and_refinement(self):
        user_prompt = "invalid_code_snippet"
        response = self.assistant.process_user_request(user_prompt)
        self.assistant.store_in_memory(response, long_term=False)
        stm_state = self.assistant.retrieve_memory(long_term=False)
        self.assertIsNotNone(stm_state, "Short-term memory should have new context after failing snippet refinement.")

    def test_physics_reset(self):
        # 1. Store initial state
        initial_weights = [
            {
                'weight': w['weight'].clone().detach(),
                'bias': w['bias'].clone().detach() if w['bias'] is not None else None
            }
            for w in self.network.get_weights()
        ]
        
        # 2. Modify weights significantly
        with torch.no_grad():
            for layer in self.network.layers:
                if isinstance(layer, nn.Linear):
                    # Add significant modification
                    layer.weight.add_(torch.ones_like(layer.weight) * 0.2)
                    if layer.bias is not None:
                        layer.bias.add_(torch.ones_like(layer.bias) * 0.2)
        
        # 3. Verify weights changed
        pre_reset_weights = self.network.get_weights()
        weight_diffs_before = []
        for i, (init, current) in enumerate(zip(initial_weights, pre_reset_weights)):
            diff = (init['weight'] - current['weight']).abs().mean().item()
            weight_diffs_before.append(diff)
            print(f"Layer {i} diff before reset: {diff}")
        
        # 4. Apply reset
        _ = self.assistant.process_user_request("Testing reset", apply_reset=True)
        
        # 5. Check reset effect
        post_reset_weights = self.network.get_weights()
        for i, (init, post) in enumerate(zip(initial_weights, post_reset_weights)):
            diff = (init['weight'] - post['weight']).abs().mean().item()
            print(f"Layer {i} diff after reset: {diff}")
            self.assertGreater(diff, 0.0, f"Layer {i}: Weights should not be identical to initial")
            self.assertLess(diff, weight_diffs_before[i], f"Layer {i}: Reset should move weights closer to initial")

    def test_memory_store_and_retrieve(self):
        self.assistant.store_in_memory("test data", long_term=False)
        stm_data = self.assistant.retrieve_memory(long_term=False)
        self.assertIsNotNone(stm_data, "Short-term memory should have data.")
        self.assistant.store_in_memory("test data LTM", long_term=True)
        ltm_data = self.assistant.retrieve_memory(index=None, long_term=True)
        self.assertTrue(len(ltm_data) > 0, "Long-term memory should store data.")

if __name__ == '__main__':
    unittest.main()