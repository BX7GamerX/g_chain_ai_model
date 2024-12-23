import unittest
import torch
import sys
import os

# Adjust path to import the OutputDecoder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from OutputDecoder import OutputDecoder

class TestOutputDecoder(unittest.TestCase):
    def setUp(self):
        # Create a small mapping for simplicity
        self.idx_to_token = {
            0: "<PAD>",
            1: "def",
            2: "foo",
            3: "(",
            4: ")",
            5: ":",
            6: "print",
            7: "'hello'",
        }
        self.decoder = OutputDecoder(idx_to_token=self.idx_to_token)

    def test_decode_argmax(self):
        # Simulate a (3, 8) distribution: 3 tokens, vocab size 8
        neural_output = torch.tensor([
            [0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.02, 0.03],  # argmax=2 => token 'foo'
            [0.01, 0.1, 0.05, 0.3, 0.2, 0.02, 0.02, 0.02], # argmax=3 => token '('
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.03, 0.07],    # argmax=5 => token ':'
        ])
        output = self.decoder.decode_output(neural_output, method="argmax")
        self.assertEqual(output, "foo ( :")

    def test_decode_embedding(self):
        # Simulate a 1D embedding
        embedding_output = torch.randn(512)
        output = self.decoder.decode_output(embedding_output, method="embedding")
        # Returns a string of 512 float elements
        self.assertTrue(len(output.split()) == 512)

    def test_unknown_token(self):
        # Argmax = 99 not in idx_to_token => <UNK_99>
        neural_output = torch.zeros(1, 100)
        neural_output[0, 99] = 1.0
        output = self.decoder.decode_output(neural_output, method="argmax")
        self.assertIn("<UNK_99>", output)

    def test_greedy_same_as_argmax(self):
        # Should behave the same as argmax in this implementation
        neural_output = torch.tensor([
            [0.1, 0.9],  # argmax=1
            [0.6, 0.4],  # argmax=0
        ])
        output_greedy = self.decoder.decode_output(neural_output, method="greedy")
        output_argmax = self.decoder.decode_output(neural_output, method="argmax")
        self.assertEqual(output_greedy, output_argmax)

if __name__ == '__main__':
    unittest.main()