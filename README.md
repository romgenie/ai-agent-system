# AI Agent System

An extensible agent system that combines the power of language models with shell command execution and file manipulation capabilities.

## Overview

This project implements a modular AI agent system that can:
- Process natural language commands
- Execute shell commands
- Answer questions using language models
- Manipulate files

The system supports multiple modes of operation:
1. **Direct Mode**: Loads model weights from disk
2. **API Mode**: Connects to a custom API server
3. **Ollama Mode**: Uses a local Ollama server with any available model

## Project Structure

```
WorkSpace/Agent/                     # Root directory for the project
├── deploy_api_server_scripts/       # Directory for scripts to launch the API server
│   └── deploy_api_server_qwen25_72b.py  # Script to start the API server
├── local_model_weights/             # Directory for model weights (for local deployment)
├── log/                             # Directory for logs
├── requirements.txt                 # Python dependencies
├── src/                             # Source code
│   ├── agent.py                     # Agent class for task handling
│   ├── llm.py                       # LLM integration
│   └── logger.py                    # Logging setup
├── start.py                         # Main entry-point script
└── static/                          # Static assets (if applicable)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/romgenie/ai-agent-system.git
cd ai-agent-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Agent

#### Direct Mode (with local model weights)
```bash
python start.py --model-path ./local_model_weights
```

#### API Mode (connecting to a running API server)
```bash
python start.py --api-url http://localhost:8760
```

#### Ollama Mode (using models from Ollama)
```bash
python start.py --ollama --model-name llama3
```

### Command Execution

You can run the agent in two modes:

#### Interactive Mode
```bash
python start.py --ollama --model-name deepseek-coder --interactive
```

#### Single Command Mode
```bash
python start.py --ollama --model-name deepseek-coder --command "What is the current date?"
```

### Starting the API Server

If you want to run your own API server:
```bash
python deploy_api_server_scripts/deploy_api_server_qwen25_72b.py
```

## Extending the System

The modular architecture makes it easy to extend the system with new capabilities:

1. Add new LLM providers in `llm.py`
2. Implement additional command types in `agent.py`
3. Create custom endpoints in the API server

## License

MIT

## Credits

This project uses various open-source components and LLM providers:
- [Ollama](https://github.com/ollama/ollama) for local model inference
- FastAPI for the API server
- Various LLM models like deepseek-coder, Qwen, etc.