import argparse
import sys
import torch
import torch.nn as nn
from typing import Optional

from TaskEncoder import TaskEncoder
from NeuralNetworkCore import NeuralNetworkCore
from Memory import ShortTermMemory, LongTermMemory
from DynamicWeightAdapter import DynamicWeightAdapter
from FeedbackEngine import FeedbackEngine
from OutputDecoder import OutputDecoder
from AICodingAssistant import AICodingAssistant

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='AI Coding Assistant')
    parser.add_argument('--beta', type=float, default=0.5,
                       help='Reset strength parameter (0-1)')
    parser.add_argument('--model-path', type=str,
                       help='Path to load pre-trained model')
    return parser.parse_args()

def initialize_components(args: argparse.Namespace) -> AICodingAssistant:
    # Create neural network first
    layers = [
        nn.Linear(512, 768),
        nn.ReLU(),
        nn.Linear(768, 256)
    ]
    network = NeuralNetworkCore(layers)
    network.initialize_network()
    
    # Create other components
    short_mem = ShortTermMemory()
    long_mem = LongTermMemory()
    weight_adapter = DynamicWeightAdapter()
    task_encoder = TaskEncoder()
    output_decoder = OutputDecoder()
    
    # Initialize FeedbackEngine with network
    feedback_engine = FeedbackEngine(
        network=network,        # Add the missing network parameter
        short_mem=short_mem,
        long_mem=long_mem,
        weight_adapter=weight_adapter,
        task_encoder=task_encoder
    )

    # Initialize AICodingAssistant
    assistant = AICodingAssistant(
        network=network,
        short_mem=short_mem,
        long_mem=long_mem,
        weight_adapter=weight_adapter,
        task_encoder=task_encoder,
        feedback_engine=feedback_engine,
        output_decoder=output_decoder,
        beta=args.beta
    )

    return assistant

def run_interactive_loop(assistant: AICodingAssistant) -> None:
    print("AI Coding Assistant initialized. Type 'exit' to quit.")
    
    while True:
        try:
            # Get user input
            user_input = input("\nEnter your coding request: ").strip()
            
            if user_input.lower() == 'exit':
                break
            
            # Process request
            response = assistant.process_user_request(user_input)
            
            # Display response
            print("\nAssistant's response:")
            print(response)
            
            # Get feedback
            feedback = input("\nWas this helpful? (y/n): ").strip().lower()
            if feedback == 'n':
                assistant.process_user_request(user_input, apply_reset=True)
        
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

def main():
    try:
        # Initialize required components
        short_mem = ShortTermMemory()
        long_mem = LongTermMemory()
        weight_adapter = DynamicWeightAdapter()
        task_encoder = TaskEncoder()

        # FeedbackEngine is initialized in initialize_components

        # Parse command line arguments
        args = parse_arguments()
        
        # Initialize components
        assistant = initialize_components(args)
        
        # Run interactive loop
        run_interactive_loop(assistant)
    
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)
    
    print("Goodbye!")
    sys.exit(0)

if __name__ == '__main__':
    main()