import unittest
import torch
import sys
import os

# Adjust the path to import Memory correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from Memory import ShortTermMemory, LongTermMemory

class TestShortTermMemory(unittest.TestCase):
    def setUp(self):
        """Initialize a ShortTermMemory instance before each test."""
        self.short_term_memory = ShortTermMemory()

    def test_initial_memory_buffer_is_empty(self):
        """Test that the memory buffer is empty upon initialization."""
        self.assertEqual(len(self.short_term_memory.memory_buffer), 0,
                         "Memory buffer should be empty upon initialization.")

    def test_update_memory_with_tensor(self):
        """Test updating memory with a valid torch.Tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        self.short_term_memory.update_memory(tensor)
        self.assertEqual(len(self.short_term_memory.memory_buffer), 1,
                         "Memory buffer should have one tensor after update.")
        self.assertTrue(torch.equal(self.short_term_memory.memory_buffer[0], tensor),
                        "Stored tensor should match the input tensor.")

    def test_update_memory_with_non_tensor_raises_type_error(self):
        """Test that updating memory with a non-tensor raises TypeError."""
        with self.assertRaises(TypeError):
            self.short_term_memory.update_memory([1, 2, 3])

    def test_get_memory_state_when_empty(self):
        """Test that get_memory_state returns None when memory buffer is empty."""
        self.assertIsNone(self.short_term_memory.get_memory_state(),
                          "get_memory_state should return None when buffer is empty.")

    def test_get_memory_state_returns_stacked_tensor(self):
        """Test that get_memory_state returns a stacked tensor of all memory entries."""
        tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        for t in tensors:
            self.short_term_memory.update_memory(t)
        memory_state = self.short_term_memory.get_memory_state()
        expected = torch.stack(tensors, dim=0)
        self.assertTrue(torch.equal(memory_state, expected),
                        "get_memory_state should return stacked tensors.")

    def test_clear_memory(self):
        """Test that clear_memory successfully empties the memory buffer."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        self.short_term_memory.update_memory(tensor)
        self.short_term_memory.clear_memory()
        self.assertEqual(len(self.short_term_memory.memory_buffer), 0,
                         "Memory buffer should be empty after clearing.")

class TestLongTermMemory(unittest.TestCase):
    def setUp(self):
        """Initialize a LongTermMemory instance before each test."""
        self.long_term_memory = LongTermMemory()

    def test_initial_knowledge_base_is_empty(self):
        """Test that the knowledge base is empty upon initialization."""
        self.assertEqual(len(self.long_term_memory.knowledge_base), 0,
                         "Knowledge base should be empty upon initialization.")

    def test_store_memory_with_tensor(self):
        """Test storing memory with a valid torch.Tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        self.long_term_memory.store_memory(tensor)
        self.assertEqual(len(self.long_term_memory.knowledge_base), 1,
                         "Knowledge base should have one entry after storing memory.")
        self.assertTrue(torch.equal(self.long_term_memory.knowledge_base[0], tensor),
                        "Stored tensor should match the input tensor.")

    def test_store_memory_with_non_tensor_raises_type_error(self):
        """Test that storing memory with a non-tensor raises TypeError."""
        with self.assertRaises(TypeError):
            self.long_term_memory.store_memory([1, 2, 3])

    def test_retrieve_memory_all(self):
        """Test retrieving all memories from the knowledge base."""
        tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        for t in tensors:
            self.long_term_memory.store_memory(t)
        retrieved = self.long_term_memory.retrieve_memory()
        self.assertEqual(len(retrieved), 2,
                         "retrieve_memory should return all stored memories.")
        for retr, orig in zip(retrieved, tensors):
            self.assertTrue(torch.equal(retr, orig),
                            "Retrieved tensor should match the stored tensor.")

    def test_retrieve_memory_specific_index(self):
        """Test retrieving a specific memory by index."""
        tensor1 = torch.tensor([1.0, 2.0])
        tensor2 = torch.tensor([3.0, 4.0])
        self.long_term_memory.store_memory(tensor1)
        self.long_term_memory.store_memory(tensor2)
        retrieved = self.long_term_memory.retrieve_memory(index=1)
        self.assertTrue(torch.equal(retrieved, tensor2),
                        "retrieve_memory should return the correct tensor for the given index.")

    def test_retrieve_memory_invalid_index_raises_index_error(self):
        """Test that retrieving memory with an invalid index raises IndexError."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        self.long_term_memory.store_memory(tensor)
        with self.assertRaises(IndexError):
            self.long_term_memory.retrieve_memory(index=5)

    def test_clear_memory(self):
        """Test that clear_memory successfully empties the knowledge base."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        self.long_term_memory.store_memory(tensor)
        self.long_term_memory.clear_memory()
        self.assertEqual(len(self.long_term_memory.knowledge_base), 0,
                         "Knowledge base should be empty after clearing.")

if __name__ == '__main__':
    unittest.main()