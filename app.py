from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import json
import os
import pathlib
import logging
from dotenv import load_dotenv, set_key, find_dotenv

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import mcp_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get the directory where the current script is located
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
# Define the path to the config file relative to the script directory
CONFIG_FILE = SCRIPT_DIR / "mcp_config.json"
# Define the static files directory
STATIC_DIR = SCRIPT_DIR / "static"

class ChatMessage(BaseModel):
    """Represents a chat message."""
    role: str
    content: str
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    """Represents a chat request from the frontend."""
    message: str
    history: List[Dict[str, Any]] # Use Dict temporarily for Pydantic AI history format

class ConfigUpdate(BaseModel):
    """Model for configuration updates."""
    base_url: str
    api_key: str
    model_choice: str
    http_referer: Optional[str] = None

app = FastAPI(title="Pydantic AI MCP Chat API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_model() -> OpenAIModel:
    """Get the configured OpenAI model."""
    llm = os.getenv('MODEL_CHOICE', 'gpt-4o-mini')
    base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
    api_key = os.getenv('LLM_API_KEY', 'no-api-key-provided')
    provider = os.getenv('PROVIDER', 'openai').lower()
    http_referer = os.getenv('HTTP_REFERER', 'https://pydantic-ai-mcp-agent.com')

    # Check if using OpenRouter
    if 'openrouter' in base_url.lower():
        logger.info("OpenRouter detected - using OpenRouter format for API")
        
        # Set OpenRouter required headers via environment variables
        # These will be picked up by the underlying OpenAI client
        os.environ['OPENAI_API_TYPE'] = 'openrouter'
        os.environ['OPENAI_ORGANIZATION'] = http_referer  # Use as HTTP referer
        
        # For OpenRouter, we need to modify the model name to include the provider prefix
        # if it's not already included and it's an OpenAI model reference
        if '/' not in llm:
            llm = f"openai/{llm}"
            logger.info(f"Modified model name for OpenRouter: {llm}")

    logger.info(f"Using model: {llm} with base URL: {base_url}")
    return OpenAIModel(
        llm,
        base_url=base_url,
        api_key=api_key
    )

async def get_pydantic_ai_agent() -> tuple[mcp_client.MCPClient, Agent]:
    """Initialize and return the MCP client and agent."""
    logger.info("Initializing MCP client and Pydantic AI agent...")
    client = mcp_client.MCPClient()
    try:
        client.load_servers(str(CONFIG_FILE))
        tools = await client.start()
        agent = Agent(model=get_model(), tools=tools)
        logger.info("MCP client and Pydantic AI agent initialized successfully.")
        return client, agent
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}", exc_info=True)
        raise

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for chat."""
    await websocket.accept()
    logger.info("WebSocket connection accepted.")
    mcp_agent_client: Optional[mcp_client.MCPClient] = None
    
    try:
        # Initialize the agent for this connection
        logger.info("Starting agent initialization...")
        mcp_agent_client, mcp_agent = await get_pydantic_ai_agent()
        logger.info("Agent initialization complete.")
        
        # Send the tools information to the client
        logger.info("Preparing tools information...")
        tools_info = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_json_schema if hasattr(tool, 'parameters_json_schema') else None
            }
            for tool in mcp_agent_client.tools
        ]
        logger.debug(f"Tools info prepared: {json.dumps(tools_info, indent=2)}")
        
        await websocket.send_text(json.dumps({
            "type": "tools",
            "content": tools_info
        }))
        logger.info("Tools information sent to client.")
        
        while True:
            # Receive message from client
            logger.info("Waiting for client message...")
            data = await websocket.receive_text()
            logger.debug(f"Received raw data: {data}")
            
            try:
                request_data = json.loads(data)
                logger.debug(f"Parsed request data: {json.dumps(request_data, indent=2)}")
                request = ChatRequest(**request_data)
                logger.info(f"Processing message: {request.message[:100]}...")
                
                # Process the message using Pydantic AI agent
                try:
                    # Get the content of the current message as a string
                    current_msg_content = request.message
                    if isinstance(current_msg_content, dict) and 'content' in current_msg_content:
                        current_msg_content = current_msg_content['content']
                    elif not isinstance(current_msg_content, str):
                        current_msg_content = str(current_msg_content)
                    
                    logger.debug(f"Current message content: {current_msg_content}")

                    # For now, don't send any history to avoid the 'Expected code to be unreachable' error
                    logger.info("Skipping message history to avoid unreachable code error")
                    pydantic_message_history = []
                    
                    logger.info("Starting agent.run_stream...")
                    logger.debug("Preparing run_stream parameters:")
                    logger.debug(f"  Current message: {json.dumps(current_msg_content, indent=2)}")
                    logger.debug(f"  Message history length: {len(pydantic_message_history)}")
                    
                    # Run the agent with proper message formatting
                    async with mcp_agent.run_stream(
                        current_msg_content,  # Pass the content string directly
                        message_history=pydantic_message_history  # Empty history to avoid errors
                    ) as result:
                        logger.debug("Agent stream started successfully")
                        curr_message = ""
                        try:
                            logger.info("Starting stream processing...")
                            async for message_delta in result.stream_text(delta=True):
                                if not message_delta:
                                    logger.debug("Empty delta received, skipping")
                                    continue
                                    
                                logger.debug(f"Delta received: {message_delta}")
                                curr_message += message_delta
                                
                                try:
                                    await websocket.send_text(json.dumps({
                                        "type": "delta",
                                        "content": message_delta
                                    }))
                                except Exception as send_error:
                                    logger.error(f"Error sending delta: {send_error}", exc_info=True)
                                    raise
                            
                            logger.info("Stream processing complete.")
                            if curr_message:
                                logger.debug(f"Final message length: {len(curr_message)}")
                                await websocket.send_text(json.dumps({
                                    "type": "complete",
                                    "content": curr_message
                                }))
                                logger.info("Complete message sent.")
                            else:
                                logger.warning("No response generated by agent")
                                raise ValueError("No response generated by the agent")
                                
                        except asyncio.CancelledError:
                            logger.warning("Stream processing was cancelled")
                            raise
                        except Exception as stream_error:
                            logger.error(f"Error during stream processing: {stream_error}", exc_info=True)
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "content": f"Stream error: {str(stream_error)}"
                            }))
                    
                except Exception as agent_error:
                    logger.error(f"Agent error: {agent_error}", exc_info=True)
                    error_message = str(agent_error)
                    if len(error_message) > 200:
                        error_message = error_message[:200] + "..."
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "content": f"Agent error: {error_message}"
                    }))
                    
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON decode error: {json_error}", exc_info=True)
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": "Invalid message format received"
                }))
            except Exception as parse_error:
                logger.error(f"Message parsing error: {parse_error}", exc_info=True)
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": f"Failed to process message: {str(parse_error)}"
                }))
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected normally.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": f"Server error: {str(e)}"
            }))
        except Exception as send_error:
            logger.error(f"Failed to send error message: {send_error}", exc_info=True)
    finally:
        if mcp_agent_client:
            logger.info("Starting MCP client cleanup...")
            try:
                async with asyncio.timeout(5.0):
                    await mcp_agent_client.cleanup()
                logger.info("MCP client cleanup completed successfully.")
            except asyncio.TimeoutError:
                logger.warning("MCP client cleanup timed out")
            except asyncio.CancelledError:
                logger.info("MCP client cleanup was cancelled")
            except Exception as cleanup_error:
                logger.error(f"Error during MCP client cleanup: {cleanup_error}", exc_info=True)
        logger.info("WebSocket connection closed.")

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def read_root() -> FileResponse:
    """Serve the main chat page (index.html)."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        logger.error("index.html not found in static directory!")
        # Consider returning a 404 or a simple error message
        return FileResponse("path/to/error/page.html", status_code=404) # Placeholder
    return FileResponse(index_path)

@app.get("/api/config")
async def get_config():
    """Get current configuration from .env file."""
    try:
        # Reload environment variables to get latest values
        load_dotenv(override=True)
        
        return {
            "base_url": os.getenv('BASE_URL', 'https://api.openai.com/v1'),
            "api_key": os.getenv('LLM_API_KEY', ''),
            "model_choice": os.getenv('MODEL_CHOICE', 'gpt-4o-mini'),
            "http_referer": os.getenv('HTTP_REFERER', 'https://pydantic-ai-mcp-agent.com')
        }
    except Exception as e:
        logger.error(f"Error reading configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to read configuration")

@app.post("/api/config")
async def update_config(config: ConfigUpdate):
    """Update configuration in .env file."""
    try:
        env_path = find_dotenv()
        if not env_path:
            raise HTTPException(status_code=404, detail=".env file not found")

        # Update .env file
        set_key(env_path, 'BASE_URL', config.base_url)
        set_key(env_path, 'LLM_API_KEY', config.api_key)
        set_key(env_path, 'MODEL_CHOICE', config.model_choice)
        
        # Save HTTP_REFERER if provided
        if config.http_referer:
            set_key(env_path, 'HTTP_REFERER', config.http_referer)

        # Reload environment variables
        load_dotenv(override=True)
        
        return {"message": "Configuration updated successfully"}
    except Exception as e:
        logger.error(f"Error updating configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update configuration")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 