# Pydantic AI MCP Agent Chat

A versatile chat application demonstrating Pydantic AI's MCP (Multi-Call Protocol) tool integration with support for various LLM providers including OpenAI, Anthropic, Gemini, OpenRouter, and local models like Ollama.

## Overview

This project provides a web-based chat interface powered by FastAPI and WebSockets. Users can interact with different AI models, leveraging MCP tools defined in a configuration file. The application dynamically adapts to different LLM APIs and handles potential compatibility issues.

⭐ Influenced by Cole Medin's project (using basics like agent and client from the linked repository):
https://github.com/coleam00/ottomator-agents/tree/main/pydantic-ai-mcp-agent

## Features

-   **Multi-Provider LLM Support:**    
    -   Direct OpenAI API & compatible endpoints (e.g., Ollama, DeepSeek).
    -   Direct Google Gemini API (`generativelanguage.googleapis.com`).
    -   Direct Anthropic API (`api.anthropic.com`).
    -   OpenRouter proxy (with custom handling for streaming/response compatibility).
    -   Handles empty API keys for local/unauthenticated endpoints.
-   **Dynamic Configuration:** In-app modal to configure Base URL, API Key (optional), and Model Name without restarting.
-   **MCP Tool Integration:**
    -   Loads and utilizes tools defined in `mcp_config.json`.
    -   Displays available tools in a side panel.
    -   Provides notifications in the chat when tools are used by the agent.
    -   Uses `agent.run()` for potentially more stable multi-tool execution (model dependent).
-   **Web Interface:**
    -   Clean, responsive chat UI built with HTML, CSS, and vanilla JavaScript.
    -   Real-time message display.
    -   Markdown rendering for code blocks in messages.
-   **Robust Backend:**
    -   FastAPI server with WebSocket handling.
    -   `pydantic-ai` for agent logic and model interaction.
    -   Specific error handling for common API issues (e.g., OpenRouter credits/rate limits, overloaded errors).
    -   DEBUG level logging for detailed troubleshooting.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/TechNavii/pydantic-ai-mcp-agent.git
    cd pydantic-ai-mcp-agent
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    # venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure MCP Tools:**
    ```bash
    cp mcp_config.json.example mcp_config.json
    # Edit mcp_config.json to define your MCP servers/tools
    cp .env.example .env
    # Edit .env.example to define your environment variables
    ```

5.  **Configure Initial LLM via `.env`:**
    - Create a `.env` file in the root directory (or copy `.env.example` if provided).
    - Set `BASE_URL`, `LLM_API_KEY` (leave empty for local models), and `MODEL_CHOICE`. These can also be set via the UI later.
      ```dotenv
      # Example .env for OpenAI
      BASE_URL=https://api.openai.com/v1
      LLM_API_KEY=your_openai_api_key
      MODEL_CHOICE=gpt-4o-mini

      # Example .env for Ollama (running locally)
      # BASE_URL=http://localhost:11434/v1
      # LLM_API_KEY=
      # MODEL_CHOICE=llama3.1:8b
      ```

## Usage

1.  **Start the application:**
    ```bash
    python app.py
    ```

2.  **Access the UI:** Open your browser and navigate to `http://localhost:8000`.

3.  **(First Time/Optional) Configure LLM:**
    - Click the settings icon (⚙️) in the header.
    - Enter the **Base URL** for your desired LLM provider (e.g., `https://api.openai.com/v1`, `https://generativelanguage.googleapis.com`, `http://localhost:11434/v1`).
    - Enter the **API Key** if required by the provider (leave blank for local models like Ollama).
    - Enter the specific **Model Name** (e.g., `gpt-4o-mini`, `gemini-1.5-flash-latest`, `llama3.1:8b`).
    - Click "Save Changes". The application will reconnect using the new settings.

4.  **Chat:** Interact with the agent. Observe tool usage notifications and responses. Use the tools icon (🛠️) to view available MCP tools.

## Architecture

-   `app.py`: Main FastAPI application, WebSocket handler, agent logic, configuration endpoint.
-   `mcp_client.py`: Client for loading and interacting with MCP tool servers defined in `mcp_config.json`.
-   `requirements.txt`: Project dependencies.
-   `mcp_config.json`: Configuration for MCP tool servers .
-   `.env` : Environment variables for initial LLM configuration(API keys, URLs).
-   `static/`: Directory containing frontend files:
    -   `index.html`: Main chat page structure and configuration modal.
    -   `styles.css`: Combined CSS for styling.
    -   `chat.js`: Frontend JavaScript logic for WebSocket communication, UI updates, configuration handling.

## Notes on Tool Calling

-   Tool calling reliability varies significantly between LLM providers and models, especially when accessed via compatibility layers like OpenRouter.
-   OpenAI models (e.g., `gpt-4o-mini`, `gpt-4-turbo`) generally offer the most reliable tool usage, even via OpenRouter.
-   Models like Gemini and Claude might require direct API integration (as implemented here) or specific prompting for reliable multi-tool sequences.
-   Local models (like those run via Ollama) might attempt tool calls unexpectedly or fail to follow multi-step instructions correctly.
-   This application currently uses `agent.run()` which may handle sequential tool calls better internally for compatible models, but detailed reporting of *which* tools were called relies on the history provided by the model/library, which can be incomplete for certain combinations.

## License

This project is available for use under open-source terms.

## Acknowledgements

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Uses [Pydantic AI](https://docs.pydantic.ai/) for the agent framework
- Influenced by [Cole Medin's Ottomator Agents](https://github.com/coleam00/ottomator-agents) 
