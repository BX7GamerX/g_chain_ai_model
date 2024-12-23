import subprocess
import tempfile
import os
import torch
import torch.nn.functional as F

from DynamicWeightAdapter import DynamicWeightAdapter
from Memory import ShortTermMemory, LongTermMemory
from NeuralNetworkCore import NeuralNetworkCore
from TaskEncoder import TaskEncoder

class FeedbackEngine:
    """
    The FeedbackEngine implements a self-corrective loop:
    1. Evaluates code snippets by running them in a sandbox.
    2. Gathers feedback based on runtime behavior or test results.
    3. Adapts network weights (and biases) to refine subsequent outputs.
    
    Mechanically, this aligns with:
    - 'Feedback-Driven Refinement' from README_3.md
    - 'Self-Corrective Feedback Loop' from README_1.md
    - 'Physics-Inspired Adaptive' references for resets after task completion (optional).
    """

    def __init__(
        self,
        network: NeuralNetworkCore,
        short_mem: ShortTermMemory,
        long_mem: LongTermMemory,
        weight_adapter: DynamicWeightAdapter,
        task_encoder: TaskEncoder
    ):
        """
        Args:
            network (NeuralNetworkCore): The core neural network.
            short_mem (ShortTermMemory): Short-term memory reference.
            long_mem (LongTermMemory): Long-term memory reference.
            weight_adapter (DynamicWeightAdapter): Dynamic weight modulation component.
            task_encoder (TaskEncoder): Encodes tasks to embeddings.
        """
        self.network = network
        self.short_mem = short_mem
        self.long_mem = long_mem
        self.weight_adapter = weight_adapter
        self.task_encoder = task_encoder

    def evaluate_code(self, code_snippet: str):
        """
        Executes code_snippet in an isolated environment and reports the exit status/output.
        Returns a dictionary containing 'success' (bool) and 'output' (str).
        """
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp_file:
            tmp_file.write(code_snippet.encode("utf-8"))
            tmp_file.flush()
            tmp_filename = tmp_file.name

        try:
            result = subprocess.run(
                ["python", tmp_filename],
                capture_output=True,
                text=True,
                timeout=5
            )
            os.unlink(tmp_filename)  # Cleanup
        except subprocess.TimeoutExpired:
            return {"success": False, "output": "Timed out during code execution."}
        except Exception as e:
            return {"success": False, "output": str(e)}

        success = (result.returncode == 0)
        return {"success": success, "output": result.stdout if success else result.stderr}

    def generate_feedback(self, evaluation_result: dict) -> str:
        """
        Generates a textual feedback message based on the evaluation results.
        In a more complex setup, this could use a specialized model
        or rules for forming detailed error feedback.
        """
        if evaluation_result["success"]:
            return "Code executed successfully. No errors detected."
        else:
            return f"Code execution failed or returned an error:\n{evaluation_result['output']}"

    def refine_network(self, current_code: str, feedback: str):
        """
        Example method that modifies the network's weights based on feedback.
        In practice, you'd parse 'feedback', compute or retrieve a suitable modulation
        tensor, then apply it to the network.
        
        The short-term memory is updated with the feedback context,
        and you can store persistent knowledge in long-term memory if needed.
        """
        # Encode the feedback to produce a modulation vector
        encoded_feedback = self.task_encoder.encode_task(feedback).mean(dim=1)  # Changed dim from 0 to 1
        code_embedding = self.task_encoder.encode_task(current_code).mean(dim=1)  # Changed dim from 0 to 1

        # Compute a modulation tensor matching the first linear layer shape (e.g., [out_features, in_features])
        modulation_mat = self.weight_adapter.compute_modulation_tensor(encoded_feedback, code_embedding)

        # Build a list of modulation_tensors for each Linear layer
        modulation_tensors = []
        for layer in self.network.layers:
            if hasattr(layer, "weight") and layer.weight is not None:
                # Ensure modulation_mat matches layer weight dimensions
                if modulation_mat.shape != layer.weight.shape:
                    modulation_mat_resized = F.interpolate(modulation_mat.unsqueeze(0), size=layer.weight.shape, mode='bilinear').squeeze(0)
                    modulation_tensors.append(modulation_mat_resized)
                else:
                    modulation_tensors.append(modulation_mat)
    
        # Adjust weights
        self.network.adjust_weights(modulation_tensors)

        # Update short-term memory with the feedback
        mean_val = modulation_mat.mean().item()  # Define mean_val before using
        self.short_mem.update_memory(torch.tensor([mean_val]))

        # Optionally persist something in long-term memory
        # self.long_mem.store_memory(torch.tensor([mean_val]))

    def full_feedback_cycle(self, code_snippet: str) -> str:
        """
        Runs a full cycle of:
        1) Evaluate code
        2) Generate feedback
        3) Refine the network
        4) Return the feedback
        """
        eval_result = self.evaluate_code(code_snippet)
        feedback = self.generate_feedback(eval_result)
        if not eval_result["success"]:
            self.refine_network(code_snippet, feedback)
        return feedback