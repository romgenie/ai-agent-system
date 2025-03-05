#!/usr/bin/env python3
"""
LLM Interface module.
Provides an interface for interacting with the Qwen2.5-72B-Instruct model,
or with Ollama's local API, or through direct model loading.
"""

import os
import json
import time
import requests
from pathlib import Path

# Import logger
from src.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)

class QwenModel:
    """
    Interface to LLM models.
    Can operate in three modes:
    1. Direct mode: Loads model weights from disk
    2. API mode: Sends requests to a running API server
    3. Ollama mode: Sends requests to a local Ollama server
    """
    
    def __init__(self, model_path=None, api_url=None, ollama_url=None, model_name="llama3"):
        """
        Initialize the model interface.
        
        Args:
            model_path (str, optional): Path to model weights directory
            api_url (str, optional): URL of the API server, if using API mode
            ollama_url (str, optional): URL of the Ollama server (e.g., http://localhost:11434)
            model_name (str, optional): Name of the Ollama model to use (default: llama3)
        """
        self.model = None
        self.tokenizer = None
        self.api_url = api_url
        self.ollama_url = ollama_url
        self.model_name = model_name
        
        # Determine mode of operation
        if ollama_url:
            # Ollama mode
            self.mode = "ollama"
            self.ollama_url = ollama_url.rstrip('/')  # Remove any trailing slash
            logger.info(f"Initialized LLM interface in Ollama mode with URL: {self.ollama_url}, model: {model_name}")
            
            # Try to check if Ollama is available
            try:
                response = requests.get(f"{self.ollama_url}/api/version", timeout=5)
                response.raise_for_status()
                logger.info(f"Connected to Ollama version: {response.json().get('version', 'unknown')}")
            except Exception as e:
                logger.warning(f"Could not connect to Ollama: {str(e)}")
                logger.warning("Continuing anyway, will attempt to connect when needed")
                
        elif api_url:
            # API mode
            self.mode = "api"
            self.api_url = api_url
            logger.info(f"Initialized LLM interface in API mode with URL: {api_url}")
        elif model_path:
            # Direct mode - load model weights
            self.mode = "direct"
            logger.info(f"Initializing model from {model_path}")
            try:
                # This is a placeholder for actual model loading code
                # In a real implementation, you would use the appropriate 
                # library to load your specific model (e.g., transformers)
                if not os.path.exists(model_path):
                    logger.warning(f"Model path {model_path} does not exist. Using dummy model.")
                    self._init_dummy_model()
                else:
                    self._load_model_from_path(model_path)
                logger.info("Model initialization complete")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                # Initialize a dummy model for testing
                self._init_dummy_model()
        else:
            # No parameters provided
            logger.error("No mode parameters (model_path, api_url, or ollama_url) provided")
            raise ValueError("Either model_path, api_url, or ollama_url must be provided")

    def _load_model_from_path(self, model_path):
        """
        Load model weights from the given path.
        In a real implementation, this would use the appropriate library.
        
        Args:
            model_path (str): Path to model weights
        """
        try:
            # This is a placeholder for actual model loading code
            logger.info(f"Model would be loaded from {model_path}")
            self._init_dummy_model()
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            raise

    def _init_dummy_model(self):
        """Initialize a dummy model for testing"""
        logger.warning("Using dummy model for testing")
        self.model = "dummy_model"
        self.tokenizer = "dummy_tokenizer"
        
    def generate(self, prompt, max_tokens=1024, temperature=0.7, top_p=0.9):
        """
        Generate a response based on the input prompt.
        
        Args:
            prompt (str): The input text to generate a response for
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature (higher = more creative, lower = more focused)
            top_p (float): Nucleus sampling parameter
            
        Returns:
            str: The generated text response
        """
        if self.mode == "ollama":
            return self._generate_via_ollama(prompt, max_tokens, temperature, top_p)
        elif self.mode == "api":
            return self._generate_via_api(prompt, max_tokens, temperature, top_p)
        else:
            return self._generate_direct(prompt, max_tokens, temperature, top_p)
    
    def _generate_via_ollama(self, prompt, max_tokens, temperature, top_p):
        """
        Generate text by querying the Ollama API.
        
        Args:
            prompt (str): The input text
            max_tokens (int): Maximum tokens to generate
            temperature (float): Temperature parameter
            top_p (float): Top-p parameter
            
        Returns:
            str: The generated text
        """
        # First, try direct completion endpoint (older Ollama versions)
        try:
            endpoint = f"{self.ollama_url}/api/completion"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens
                }
            }
            
            logger.debug(f"Trying completion endpoint: {endpoint}")
            start_time = time.time()
            
            response = requests.post(
                endpoint,
                json=payload,
                timeout=120  # 120 second timeout for larger models
            )
            
            if response.status_code == 200:
                elapsed_time = time.time() - start_time
                logger.debug(f"Response received from Ollama completion in {elapsed_time:.2f}s")
                
                result = response.json()
                logger.debug(f"Ollama completion response: {json.dumps(result)}")
                
                if "response" in result:
                    return result["response"]
                else:
                    logger.error(f"Unexpected Ollama API completion response format: {result}")
            else:
                logger.warning(f"Completion endpoint failed with status {response.status_code}, trying chat endpoint")
        except Exception as e:
            logger.warning(f"Error with completion endpoint, trying chat endpoint: {str(e)}")
            
        # If that fails, try the chat endpoint (newer Ollama versions)
        try:
            endpoint = f"{self.ollama_url}/api/chat"
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens
                }
            }
            
            logger.debug(f"Trying chat endpoint: {endpoint}")
            start_time = time.time()
            
            response = requests.post(
                endpoint,
                json=payload,
                timeout=120  # 120 second timeout for larger models
            )
            
            if response.status_code == 200:
                elapsed_time = time.time() - start_time
                logger.debug(f"Response received from Ollama chat in {elapsed_time:.2f}s")
                
                result = response.json()
                logger.debug(f"Ollama chat response: {json.dumps(result)}")
                
                if "message" in result and "content" in result["message"]:
                    return result["message"]["content"]
                else:
                    logger.error(f"Unexpected Ollama API chat response format: {result}")
            else:
                logger.warning(f"Chat endpoint failed with status {response.status_code}")
                
            # Both methods failed but we got some response, return its content
            return f"Error: Could not get a valid response from Ollama API. Status code: {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {str(e)}")
            return f"Error communicating with Ollama API server: {str(e)}"
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Ollama API response: {str(e)}")
            return "Error parsing response from Ollama API server"
        except Exception as e:
            logger.error(f"Unexpected error in Ollama API request: {str(e)}")
            return f"Unexpected error with Ollama: {str(e)}"
    
    def _generate_via_api(self, prompt, max_tokens, temperature, top_p):
        """
        Generate text by querying the API server.
        
        Args:
            prompt (str): The input text
            max_tokens (int): Maximum tokens to generate
            temperature (float): Temperature parameter
            top_p (float): Top-p parameter
            
        Returns:
            str: The generated text
        """
        try:
            endpoint = f"{self.api_url}/query"
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            
            logger.debug(f"Sending request to {endpoint}")
            start_time = time.time()
            
            response = requests.post(
                endpoint,
                json=payload,
                timeout=60  # 60 second timeout
            )
            
            elapsed_time = time.time() - start_time
            logger.debug(f"Response received in {elapsed_time:.2f}s")
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse JSON response
            result = response.json()
            
            if "response" not in result:
                logger.error(f"Unexpected API response format: {result}")
                return "Error: Unexpected response format from API"
                
            return result["response"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return f"Error communicating with API server: {str(e)}"
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {str(e)}")
            return "Error parsing response from API server"
        except Exception as e:
            logger.error(f"Unexpected error in API request: {str(e)}")
            return f"Unexpected error: {str(e)}"
    
    def _generate_direct(self, prompt, max_tokens, temperature, top_p):
        """
        Generate text using the locally loaded model.
        
        Args:
            prompt (str): The input text
            max_tokens (int): Maximum tokens to generate
            temperature (float): Temperature parameter
            top_p (float): Top-p parameter
            
        Returns:
            str: The generated text
        """
        # Check if model is loaded
        if self.model == "dummy_model":
            logger.warning("Using dummy model for generation")
            return f"This is a dummy response to: {prompt}"
            
        try:
            # This is a placeholder for actual model inference code
            logger.info(f"Model would generate text with params: max_tokens={max_tokens}, temp={temperature}")
            return f"This is a simulated response to: {prompt}"
            
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            return f"Error generating text: {str(e)}"