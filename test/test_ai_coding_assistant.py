import unittest
import sys
import os

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
            initial_weights=initial_weights,
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
        # Provide a prompt and request a reset afterwards
        user_prompt = "Testing partial reset."
        _ = self.assistant.process_user_request(user_prompt, apply_reset=True)
        # Check if the weights were partially reset
        current_weights = self.network.get_weights()
        # We expect partial difference from initial
        for init_dict, curr_dict in zip(self.assistant.physics_reset.initial_weights, current_weights):
            diff = (init_dict['weight'] - curr_dict['weight']).abs().mean().item()
            self.assertGreater(diff, 0.0, "Weights should not be identical to initial if beta < 1.")
            self.assertLess(diff, 0.5, "Weights should be partially, not fully reset.")

    def test_memory_store_and_retrieve(self):
        self.assistant.store_in_memory("test data", long_term=False)
        stm_data = self.assistant.retrieve_memory(long_term=False)
        self.assertIsNotNone(stm_data, "Short-term memory should have data.")
        self.assistant.store_in_memory("test data LTM", long_term=True)
        ltm_data = self.assistant.retrieve_memory(index=None, long_term=True)
        self.assertTrue(len(ltm_data) > 0, "Long-term memory should store data.")

if __name__ == '__main__':
    unittest.main()