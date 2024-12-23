import unittest
import torch
import sys
import os
import tempfile

# Adjust the path to import modules correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from FeedbackEngine import FeedbackEngine
from NeuralNetworkCore import NeuralNetworkCore
from DynamicWeightAdapter import DynamicWeightAdapter
from Memory import ShortTermMemory, LongTermMemory
from TaskEncoder import TaskEncoder
import torch.nn as nn

class TestFeedbackEngine(unittest.TestCase):
    def setUp(self):
        # Create minimal NeuralNetworkCore
        layers = [
            nn.Linear(512, 768),
            nn.ReLU(),
            nn.Linear(768, 256)
        ]
        self.network_core = NeuralNetworkCore(layers)
        self.network_core.initialize_network()

        # Create memory modules
        self.short_mem = ShortTermMemory()
        self.long_mem = LongTermMemory()

        # Create adapter and encoder
        self.weight_adapter = DynamicWeightAdapter()
        self.task_encoder = TaskEncoder()

        # Instantiate FeedbackEngine
        self.engine = FeedbackEngine(
            network=self.network_core,
            short_mem=self.short_mem,
            long_mem=self.long_mem,
            weight_adapter=self.weight_adapter,
            task_encoder=self.task_encoder
        )

    def test_evaluate_code_success(self):
        code_snippet = "print('Hello, world!')"
        result = self.engine.evaluate_code(code_snippet)
        self.assertTrue(result["success"], "Code snippet should execute successfully.")
        self.assertIn("Hello, world!", result["output"], "Output should contain the printed text.")

    def test_evaluate_code_failure(self):
        code_snippet = "raise ValueError('This is an error')"
        result = self.engine.evaluate_code(code_snippet)
        self.assertFalse(result["success"], "Code snippet should fail.")
        self.assertIn("ValueError: This is an error", result["output"], "Output should contain the error message.")

    def test_generate_feedback_success(self):
        eval_result = {"success": True, "output": "All good!"}
        feedback = self.engine.generate_feedback(eval_result)
        self.assertIn("No errors detected", feedback)

    def test_generate_feedback_failure(self):
        eval_result = {"success": False, "output": "Some error occurred."}
        feedback = self.engine.generate_feedback(eval_result)
        self.assertIn("Some error occurred.", feedback)

    def test_refine_network(self):
        code_snippet = "print('Test')"
        feedback = "Code executed successfully."
        self.engine.refine_network(code_snippet, feedback)
        # Add assertions to verify weight adjustments
        current_weights = self.engine.network.get_weights()
        self.assertIsNotNone(current_weights)
        # ...additional assertions...

    def test_full_feedback_cycle(self):
        bad_code = "print('Hello World'"  # Missing closing parenthesis
        feedback = self.engine.full_feedback_cycle(bad_code)
        # Add assertions or checks as needed
        self.assertIn("failed", feedback.lower())

if __name__ == '__main__':
    unittest.main()