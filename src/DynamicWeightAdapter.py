import torch
import torch.nn.functional as F

class DynamicWeightAdapter:
    def __init__(self):
        """
        Initializes the DynamicWeightAdapter.
        """
        pass

    def compute_modulation_tensor(self, query, key):
        """
        Computes the modulation tensor M using the query and key embeddings.

        Args:
            query (torch.Tensor): The query tensor representing the task-specific input.
                                   Shape: (d, ) where d is the embedding dimension.
            key (torch.Tensor): The key tensor representing contextual information.
                                 Shape: (d, )

        Returns:
            torch.Tensor: The modulation tensor M with the same shape as the weights.
                          Shape: (d, d)
        """
        # Ensure query and key are 2D tensors
        if query.dim() == 1:
            query = query.unsqueeze(0)  # Shape: (1, d)
        if key.dim() == 1:
            key = key.unsqueeze(0)      # Shape: (1, d)
    
        # Check dimensions
        assert query.size(1) == key.size(1), "Query and Key must have the same embedding dimension."
    
        modulation_raw = torch.matmul(query.t(), key)  # Shape: (d, d)
        modulation = F.softmax(modulation_raw, dim=1)  # Shape: (d, d)
    
        return modulation

    def apply_modulation(self, weights, modulation_tensor):
        """
        Applies the modulation tensor to the given weights via element-wise multiplication.

        Args:
            weights (torch.Tensor): The original weights of a Linear layer.
                                     Shape: (out_features, in_features)
            modulation_tensor (torch.Tensor): The modulation tensor M.
                                             Shape: (out_features, in_features)

        Returns:
            torch.Tensor: The modulated weights.
        """
        if weights.shape != modulation_tensor.shape:
            raise ValueError("Shape of weights and modulation_tensor must match.")

        # Apply element-wise multiplication
        modulated_weights = weights * modulation_tensor

        return modulated_weights