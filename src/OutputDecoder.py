import torch

class OutputDecoder:
    """
    The OutputDecoder converts the final network output (a tensor) into a string
    representation of code. It supports different decoding strategies:

    1) Vector-to-Token Mapping:
       - Interprets each row of the neural output as a probability distribution over a vocabulary.
       - Selects tokens via argmax or sampling, then concatenates them into a code snippet.

    2) Direct Vector Interpretation:
       - Treats the neural output as a continuous embedding for code generation (e.g., if a separate
         code-generation model is used). This can be further processed or projected into text tokens.

    In practice, adapt the decoding process to match the shape and meaning of your final layer's output.
    Refer to the relevant README docs for details on how the final layer encodes code representations.
    """

    def __init__(self, idx_to_token=None):
        """
        Args:
            idx_to_token (dict or list, optional): A mapping from token indices to string tokens.
                                                  Required if using vector-to-token decoding.
        """
        self.idx_to_token = idx_to_token if idx_to_token is not None else {}

    def decode_output(self, neural_output: torch.Tensor, method: str = "argmax") -> str:
        """
        Decodes the neural network output to a code snippet string based on the selected method.

        Args:
            neural_output (torch.Tensor): The output tensor from the network.
              - Shape could be (sequence_length, vocab_size) for token distributions,
                or simply (m,) for a continuous representation.
            method (str): Which decoding method to use. Supported:
              - "argmax": Select the token ID with the highest probability at each step.
              - "greedy": Same as argmax in this simple example.
              - "embedding": Interpret the output as a continuous embedding (placeholder).

        Returns:
            str: The decoded code snippet.
        """
        if method == "argmax":
            # Expect shape (sequence_length, vocab_size)
            if neural_output.dim() == 2:
                # Argmax across the last dimension => token IDs
                token_ids = torch.argmax(neural_output, dim=-1).tolist()
                tokens = [self.idx_to_token.get(tid, f"<UNK_{tid}>") for tid in token_ids]
                return " ".join(tokens)
            else:
                # If it's not a 2D distribution, treat it as a single embedding or unknown format
                return self._decode_embedding(neural_output)

        elif method == "greedy":
            # Identical to argmax in this basic example
            if neural_output.dim() == 2:
                token_ids = torch.argmax(neural_output, dim=-1).tolist()
                tokens = [self.idx_to_token.get(tid, f"<UNK_{tid}>") for tid in token_ids]
                return " ".join(tokens)
            else:
                return self._decode_embedding(neural_output)

        elif method == "embedding":
            # Interpret the vector as a continuous embedding; placeholder
            return self._decode_embedding(neural_output)

        else:
            raise ValueError(f"Unsupported decoding method: {method}")

    def _decode_embedding(self, embedding: torch.Tensor) -> str:
        """
        Placeholder for decoding a continuous embedding. In practice:
        - Use a specialized model to convert embeddings back to code,
        - Or apply a transform that maps embeddings to tokens.

        For now, returns a string representation of the embedding.
        """
        # Example: flatten the embedding and convert to string
        flattened = embedding.flatten().tolist()
        return " ".join([f"{val:.4f}" for val in flattened])