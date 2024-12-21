from transformers import BertModel, BertTokenizer
import torch

class TaskEncoder:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def encode_task(self, query):
        inputs = self.tokenizer(query, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def get_vector_representation(self, encoded_task):
        return torch.mean(encoded_task, dim=1).squeeze().tolist()