from transformers import BertModel, BertTokenizer
import torch

class TaskEncoder:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    
    def encode_task(self, task_description: str) -> torch.Tensor:
        """
        Encode a task description into a tensor representation.
    
        Args:
            task_description (str): The task to encode
    
        Returns:
            torch.Tensor: Shape (batch_size=1, sequence_length, hidden_size)
        """
        # Tokenize and encode
        encoded = self.tokenizer(
            task_description,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
    
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**encoded)
    
        # Get last hidden state
        last_hidden_state = outputs.last_hidden_state  # Shape: (1, sequence_length, hidden_size)
    
        return last_hidden_state

    def get_vector_representation(self, tensor: torch.Tensor) -> list:
        # Convert tensor to list and compute mean across appropriate dimension
        if tensor.dim() > 1:
            # If tensor has multiple dimensions, compute mean across appropriate axis
            return tensor.mean(dim=0).tolist()
        return tensor.tolist()
    
    def get_task_embedding(self, task: str) -> list:
        encoded = self.encode_task(task)
        return self.get_vector_representation(encoded)