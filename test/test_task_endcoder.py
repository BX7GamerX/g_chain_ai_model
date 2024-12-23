import unittest
from TaskEncoder import TaskEncoder
import torch

class TestTaskEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = TaskEncoder()
    
    def test_encode_task(self):
        query = "Test the TaskEncoder class."
        encoded_task = self.encoder.encode_task(query)
        self.assertIsNotNone(encoded_task)
        self.assertEqual(encoded_task.dim(), 3)  # (batch_size, sequence_length, hidden_size)
    
    def test_get_vector_representation(self):
        # Create a dummy tensor
        encoded_task = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # shape (1,2,3)
        vector = self.encoder.get_vector_representation(encoded_task)
        expected_vector = [2.5, 3.5, 4.5]
        self.assertEqual(vector, expected_vector)
    
    def test_full_process(self):
        query = "Integrate the TaskEncoder functionality."
        encoded_task = self.encoder.encode_task(query)
        vector = self.encoder.get_vector_representation(encoded_task)
        self.assertIsInstance(vector, list)
        self.assertEqual(len(vector), encoded_task.size(2))  # hidden_size

if __name__ == '__main__':
    unittest.main()