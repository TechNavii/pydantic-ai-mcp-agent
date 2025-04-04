# Pydantic AI MCP Agent

A chat application with Pydantic AI MCP (Multi-Call Protocol) tool integration.

## Overview

This project implements a chat interface that allows users to interact with AI models using Pydantic AI's MCP tools. The application provides a web-based interface where users can send messages and receive responses from the AI, with the AI able to use various MCP tools to accomplish tasks.

 ⭐︎Infludenced from Cole Medin`s project. (using basics like agent and client from below repository) 
 https://github.com/coleam00/ottomator-agents/tree/main/pydantic-ai-mcp-agent

## Features

- WebSocket-based chat interface
- Integration with Pydantic AI's MCP tools
- Real-time streaming of AI responses
- Clean and responsive UI

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/TechNavii/pydantic-ai-mcp-agent.git
   cd pydantic-ai-mcp-agent
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your configuration:
   ```bash
   cp mcp_config.json.example mcp_config.json
   # Edit mcp_config.json to add your API keys and other configuration
   ```

## Usage

1. Start the application:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to http://localhost:8000

3. Start chatting with the AI agent!

## Architecture

- `app.py`: Main application file with FastAPI server
- `mcp_client.py`: Client for interacting with MCP tools
- `pydantic_mcp_agent.py`: Implementation of the Pydantic AI agent
- `static/`: Frontend files (HTML, CSS, JavaScript)

## Configuration

The `mcp_config.json` file should contain your API keys and other configuration parameters. Use the provided example as a template.

## License

This project is available for use under open-source terms.

## Acknowledgements

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Uses [Pydantic AI](https://docs.pydantic.ai/) for type checking and validation 