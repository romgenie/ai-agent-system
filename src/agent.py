#!/usr/bin/env python3
"""
Agent module.
Defines the main Agent class which orchestrates system operations, processes commands,
and coordinates interactions with the LLM.
"""

import os
import sys
import json
import time
import re
import subprocess
from pathlib import Path

# Import project modules
from src.llm import QwenModel
from src.logger import setup_logger

# Setup logger
logger = setup_logger(__name__, level="DEBUG")

class Agent:
    """
    Agent class that serves as the central coordinator of the AI system.
    Processes user commands, interacts with the LLM, and executes system operations.
    """
    
    def __init__(self, api_url=None, model_path=None, ollama_url=None, model_name="deepseek-coder"):
        """
        Initialize the Agent with LLM connection.
        
        Args:
            api_url (str, optional): URL of the LLM API server
            model_path (str, optional): Path to model weights (for direct model loading)
            ollama_url (str, optional): URL of the Ollama server
            model_name (str, optional): Name of the Ollama model to use
        """
        logger.info("Initializing Agent")
        
        # Initialize LLM interface
        try:
            self.llm = QwenModel(
                model_path=model_path, 
                api_url=api_url,
                ollama_url=ollama_url,
                model_name=model_name
            )
            logger.info("LLM interface initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM interface: {str(e)}")
            raise
        
        # Command history
        self.command_history = []
        
        # Agent state
        self.running = False
        
        logger.info("Agent initialization complete")
    
    def start(self):
        """Start the agent"""
        logger.info("Starting Agent")
        self.running = True
        
    def stop(self):
        """Stop the agent"""
        logger.info("Stopping Agent")
        self.running = False
    
    def process_command(self, command):
        """
        Process a user command.
        
        Args:
            command (str): The command string to process
            
        Returns:
            dict: Response with status and content
        """
        logger.info(f"Processing command: {command}")
        
        # Add to command history
        self.command_history.append(command)
        
        try:
            # First, simply try to interpret as "shell_command" the most common commands
            if command.startswith("ls") or command.startswith("cd ") or command.startswith("pwd") or \
               command.startswith("cat ") or command.startswith("date") or command.startswith("echo "):
                action = {"type": "shell_command", "command": command}
                logger.debug(f"Direct shell command interpretation: {action}")
            else:
                # Generate system prompt with the command
                prompt = self._create_system_prompt(command)
                
                # Get response from LLM
                llm_response = self.llm.generate(
                    prompt=prompt,
                    max_tokens=1024,
                    temperature=0.7
                )
                
                logger.debug(f"LLM response: {llm_response}")
                
                # Parse LLM response to determine action
                action = self._parse_llm_response(llm_response)
                logger.debug(f"Parsed action: {action}")
            
            # Execute the determined action
            result = self._execute_action(action)
            
            logger.info(f"Command processed successfully: {command}")
            return {
                "status": "success",
                "action": action.get("type", "response"),
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error processing command '{command}': {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _create_system_prompt(self, command):
        """
        Create a system prompt for the LLM with the user command.
        
        Args:
            command (str): The user command
            
        Returns:
            str: The formatted system prompt
        """
        # Simple prompt template
        prompt = f"""
You are helping process user commands in a command-line interface.

User command: "{command}"

Based on this command, do one of these:
1. If it's a system command (like checking time, listing files, etc.), reply with:
   ACTION: shell
   COMMAND: <the exact shell command to run>

2. If it's a question or conversation, reply with:
   ACTION: respond
   CONTENT: <your helpful response>

3. If there's an error or you can't process it, reply with:
   ACTION: error
   REASON: <explanation of the error>

Reply using ONLY this format, with no additional text.
"""
        return prompt
    
    def _parse_llm_response(self, response):
        """
        Parse the LLM's response to extract the action.
        
        Args:
            response (str): The LLM response text
            
        Returns:
            dict: The parsed action
        """
        try:
            # First try to parse using the simple format
            # Look for ACTION: <type> followed by the appropriate field
            shell_match = re.search(r'ACTION:\s*shell\s*\nCOMMAND:\s*(.*?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
            if shell_match:
                command = shell_match.group(1).strip()
                return {"type": "shell_command", "command": command}
                
            respond_match = re.search(r'ACTION:\s*respond\s*\nCONTENT:\s*(.*?)(?:\n\n|$)', response, re.IGNORECASE | re.DOTALL)
            if respond_match:
                content = respond_match.group(1).strip()
                return {"type": "response", "content": content}
                
            error_match = re.search(r'ACTION:\s*error\s*\nREASON:\s*(.*?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
            if error_match:
                error = error_match.group(1).strip()
                return {"type": "error", "error": error}
                
            # If we couldn't parse using our simple format, treat as a regular response
            logger.warning("Could not parse LLM response as ACTION format, treating as plain text")
            return {
                "type": "response",
                "content": response
            }
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return {
                "type": "error",
                "error": f"Failed to parse LLM response: {str(e)}",
                "original_response": response
            }
    
    def _execute_action(self, action):
        """
        Execute the parsed action.
        
        Args:
            action (dict): The action to execute
            
        Returns:
            dict: The result of the action
        """
        action_type = action.get("type", "unknown")
        
        if action_type == "shell_command":
            # Clean up the command if needed
            command = action.get("command", "").strip()
            return self._execute_shell_command(command)
            
        elif action_type == "response":
            # Just return the content
            return {"output": action.get("content", "")}
            
        elif action_type == "error":
            logger.error(f"Action error: {action.get('error', 'Unknown error')}")
            return {"error": action.get("error", "Unknown error")}
            
        else:
            logger.warning(f"Unknown action type: {action_type}")
            return {"error": f"Unknown action type: {action_type}"}
    
    def _execute_shell_command(self, command):
        """
        Execute a shell command.
        
        Args:
            command (str): The shell command to execute
            
        Returns:
            dict: The result with stdout, stderr, and return code
        """
        if not command:
            return {"error": "Empty command"}
        
        logger.info(f"Executing shell command: {command}")
        
        try:
            # Execute the command
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Set a timeout (30 seconds)
            try:
                stdout, stderr = process.communicate(timeout=30)
                return_code = process.returncode
                
                result = {
                    "stdout": stdout,
                    "stderr": stderr,
                    "return_code": return_code
                }
                
                if return_code != 0:
                    logger.warning(f"Command returned non-zero exit code {return_code}: {command}")
                    result["status"] = "error"
                else:
                    result["status"] = "success"
                    
                return result
                
            except subprocess.TimeoutExpired:
                # Kill the process if it times out
                process.kill()
                logger.error(f"Command timed out after 30 seconds: {command}")
                return {
                    "status": "error",
                    "error": "Command execution timed out after 30 seconds"
                }
                
        except Exception as e:
            logger.error(f"Error executing command '{command}': {str(e)}")
            return {
                "status": "error",
                "error": f"Command execution failed: {str(e)}"
            }