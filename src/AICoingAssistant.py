import torch

from TaskEncoder import TaskEncoder
from NeuralNetworkCore import NeuralNetworkCore
from Memory import ShortTermMemory, LongTermMemory
from DynamicWeightAdapter import DynamicWeightAdapter
from FeedbackEngine import FeedbackEngine
from OutputDecoder import OutputDecoder
from PhysicsInspiredReset import PhysicsInspiredReset

class AICodingAssistant:
    """
    A high-level orchestrator that uses all modules to:
    1) Encode user instructions via TaskEncoder.
    2) Generate or refine code snippets using NeuralNetworkCore and OutputDecoder.
    3) Manage ephemeral data via ShortTermMemory and deeper knowledge via LongTermMemory.
    4) Dynamically adjust weights with DynamicWeightAdapter based on context or user feedback.
    5) Evaluate code snippets with FeedbackEngine, then refine or reset network.
    6) Optionally apply a PhysicsInspiredReset after detected task completion.
    """

    def __init__(
        self,
        network: NeuralNetworkCore,
        short_mem: ShortTermMemory,
        long_mem: LongTermMemory,
        weight_adapter: DynamicWeightAdapter,
        task_encoder: TaskEncoder,
        feedback_engine: FeedbackEngine,
        output_decoder: OutputDecoder,
        initial_weights=None,
        beta=0.5
    ):
        """
        Args:
            network (NeuralNetworkCore): Core network.
            short_mem (ShortTermMemory): Short-term memory instance.
            long_mem (LongTermMemory): Long-term memory instance.
            weight_adapter (DynamicWeightAdapter): Dynamic weight adapter for context-based modulations.
            task_encoder (TaskEncoder): BERT-based (or similar) text-to-embedding encoder.
            feedback_engine (FeedbackEngine): Method for evaluating code & refining weights based on feedback.
            output_decoder (OutputDecoder): Decodes network outputs into code strings.
            initial_weights (list of dict, optional): For optional PhysicsInspiredReset usage.
            beta (float): Reset strength factor for PhysicsInspiredReset.
        """
        self.network = network
        self.short_mem = short_mem
        self.long_mem = long_mem
        self.weight_adapter = weight_adapter
        self.task_encoder = task_encoder
        self.feedback_engine = feedback_engine
        self.output_decoder = output_decoder

        # Initialize optional physics-inspired resetting
        self.physical_resetter = None
        if initial_weights is not None:
            self.physical_resetter = PhysicsInspiredReset(initial_weights, beta=beta)

    def process_user_request(self, user_prompt: str, apply_reset: bool = False) -> str:
        """
        Main entry point to:
        1) Encode user prompt into embeddings.
        2) Pass through NeuralNetworkCore -> produce output tensor.
        3) Decode with OutputDecoder.
        4) Optionally apply code evaluation+feedback loop.

        Args:
            user_prompt (str): The user's request.
            apply_reset (bool): If True, applies partial reset after generation.

        Returns:
            str: The generated or refined code snippet.
        """
        # 1) Encode request
        encoded_prompt = self.task_encoder.encode_task(user_prompt)
        embedding_vector = torch.mean(encoded_prompt, dim=0)

        # 2) Forward pass
        output_tensor = self.network.forward_propagation(embedding_vector)

        # 3) Decode to code snippet
        code_snippet = self.output_decoder.decode_output(output_tensor, method="argmax")

        # 4) Optionally evaluate code snippet and refine
        evaluation = self.feedback_engine.evaluate_code(code_snippet)
        feedback = self.feedback_engine.generate_feedback(evaluation)

        # If code fails, refine network
        if not evaluation["success"]:
            self.feedback_engine.refine_network(code_snippet, feedback)

        # Optionally apply physics-inspired reset
        if apply_reset and self.physical_resetter is not None:
            current_weights = self.network.get_weights()
            new_state = self.physical_resetter.reset_parameters(current_weights)
            self.network.set_weights(new_state)

        return code_snippet

    def store_in_memory(self, data: str, long_term: bool = False):
        """
        Convenience method to store data in memory modules.
        """
        tensor_data = torch.tensor([ord(c) for c in data], dtype=torch.float)
        if long_term:
            self.long_mem.store_memory(tensor_data)
        else:
            self.short_mem.update_memory(tensor_data)

    def retrieve_memory(self, index=None, long_term: bool = False):
        """
        Convenience method to retrieve data from memory modules.
        """
        if long_term:
            return self.long_mem.retrieve_memory(index=index)
        else:
            stm_state = self.short_mem.get_memory_state()
            return stm_state if stm_state is not None else None