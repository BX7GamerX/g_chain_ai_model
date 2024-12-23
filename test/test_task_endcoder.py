import unittest
from TaskEncoder import TaskEncoder
import torch

class TestTaskEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = TaskEncoder()
    
    def test_encode_task(self):
        task = "Test task"
        encoded_task = self.encoder.encode_task(task)
        self.assertEqual(encoded_task.dim(), 3)  # (batch_size, sequence_length, hidden_size)
    
    def test_get_vector_representation(self):
        input_tensor = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        vector = self.encoder.get_vector_representation(input_tensor)
        expected_vector = [2.5, 3.5, 4.5]  # Mean across first dimension
        self.assertEqual(vector, expected_vector)
    
    def test_full_process(self):
        task = "Test task"
        vector = self.encoder.get_task_embedding(task)
        self.assertIsInstance(vector, list)

if __name__ == '__main__':
    unittest.main()