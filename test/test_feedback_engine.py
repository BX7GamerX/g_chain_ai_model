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
        # Provide a dummy code snippet and feedback
        code_snippet = "def foo():\n    return 'bar'"
        feedback = "Fix your code"

        # Capture initial weights for comparison
        initial_weights = self.network_core.get_weights()
        self.engine.refine_network(code_snippet, feedback)

        # Expect that weights have changed
        new_weights = self.network_core.get_weights()
        for init_w, new_w in zip(initial_weights, new_weights):
            self.assertFalse(torch.allclose(init_w['weight'], new_w['weight']), "Weights should be different after refinement.")

        # Check short-term memory updated
        stm_state = self.short_mem.get_memory_state()
        self.assertIsNotNone(stm_state, "Short-term memory should hold the new feedback context.")

    def test_full_feedback_cycle(self):
        # Code that fails
        bad_code = "raise KeyError('Fail!')"
        feedback = self.engine.full_feedback_cycle(bad_code)
        self.assertIn("Fail!", feedback, "Feedback should mention the error.")
        # Check short-term memory has new data
        stm_state = self.short_mem.get_memory_state()
        self.assertIsNotNone(stm_state, "Short-term memory should hold context after the feedback cycle.")

        # Code that succeeds
        good_code = "print('All good!')"
        feedback2 = self.engine.full_feedback_cycle(good_code)
        self.assertIn("No errors detected", feedback2, "Feedback should indicate a successful run.")

if __name__ == '__main__':
    unittest.main()