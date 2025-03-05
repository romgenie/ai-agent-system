#!/usr/bin/env python3
"""
Main entry point for the AI system.
Initializes the Agent and starts the primary process.
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Import project modules
from src.agent import Agent
from src.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Start the AI Agent system")
    
    # Model source options - only one should be used
    model_group = parser.add_argument_group('Model Source (use only one)')
    
    model_group.add_argument(
        "--api-url",
        type=str,
        help="URL of the LLM API server",
        default=None
    )
    
    model_group.add_argument(
        "--model-path",
        type=str,
        help="Path to local model weights",
        default="./local_model_weights"
    )
    
    model_group.add_argument(
        "--ollama",
        action="store_true",
        help="Use Ollama for LLM inference"
    )
    
    # Ollama specific options
    ollama_group = parser.add_argument_group('Ollama Options')
    
    ollama_group.add_argument(
        "--ollama-url",
        type=str,
        help="URL of the Ollama server (default: http://localhost:11434)",
        default="http://localhost:11434"
    )
    
    ollama_group.add_argument(
        "--model-name",
        type=str,
        help="Name of the Ollama model to use (default: llama3)",
        default="llama3"
    )
    
    # Operation mode
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start in interactive mode to accept commands directly"
    )
    
    parser.add_argument(
        "--command",
        type=str,
        help="Single command to execute (non-interactive mode)",
        default=None
    )
    
    return parser.parse_args()

def start_interactive_mode(agent):
    """
    Start the agent in interactive mode, accepting commands from the console.
    
    Args:
        agent (Agent): The initialized Agent instance
    """
    logger.info("Starting interactive mode")
    print("\nAI Agent Interactive Mode")
    print("Type 'exit' or 'quit' to end the session")
    print("------------------------------")
    
    agent.start()
    
    while agent.running:
        try:
            # Get command from user
            command = input("\nEnter command: ")
            
            # Check for exit command
            if command.lower() in ["exit", "quit"]:
                agent.stop()
                print("Exiting interactive mode")
                break
                
            # Process the command
            result = agent.process_command(command)
            
            # Display the result
            if result["status"] == "success":
                if "result" in result and "output" in result["result"]:
                    print("\nResult:")
                    print(result["result"]["output"])
                elif "result" in result and "stdout" in result["result"]:
                    print("\nCommand Output:")
                    if result["result"]["stdout"]:
                        print(result["result"]["stdout"])
                    if result["result"]["stderr"]:
                        print("Errors:", result["result"]["stderr"])
                    print(f"Return code: {result['result']['return_code']}")
                else:
                    print("\nCommand executed successfully")
            else:
                print(f"\nError: {result.get('error', 'Unknown error')}")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            agent.stop()
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {str(e)}")
            print(f"\nSystem error: {str(e)}")

def main():
    """Main entry point"""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Display startup banner
        print("\n===================================")
        print("  AI Agent System")
        print("===================================\n")
        
        # Determine LLM mode
        if args.ollama:
            # Ollama mode
            logger.info(f"Using Ollama mode with model: {args.model_name}")
            agent = Agent(
                ollama_url=args.ollama_url,
                model_name=args.model_name
            )
        elif args.api_url:
            # API mode
            logger.info(f"Using API mode with URL: {args.api_url}")
            agent = Agent(api_url=args.api_url)
        else:
            # Direct mode
            logger.info(f"Using direct mode with model path: {args.model_path}")
            agent = Agent(model_path=args.model_path)
            
        logger.info("Agent initialized successfully")
        
        # Determine operating mode
        if args.command:
            # Single command mode
            logger.info(f"Executing single command: {args.command}")
            result = agent.process_command(args.command)
            
            if result["status"] == "success":
                if "result" in result and "output" in result["result"]:
                    print("\nResult:")
                    print(result["result"]["output"])
                elif "result" in result and "stdout" in result["result"]:
                    print("\nCommand Output:")
                    if result["result"]["stdout"]:
                        print(result["result"]["stdout"])
                    if result["result"]["stderr"]:
                        print("Errors:", result["result"]["stderr"])
                    print(f"Return code: {result['result']['return_code']}")
                else:
                    print("\nCommand executed successfully")
            else:
                print(f"\nError: {result.get('error', 'Unknown error')}")
                
        elif args.interactive:
            # Interactive mode
            start_interactive_mode(agent)
        else:
            # Default to interactive mode if no specific mode is specified
            start_interactive_mode(agent)
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"Fatal error: {str(e)}")
        return 1
        
    logger.info("System shutdown complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())