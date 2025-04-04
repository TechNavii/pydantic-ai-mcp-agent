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

    # Check if using OpenRouter
    if 'openrouter' in base_url.lower():
        logger.info("OpenRouter detected - configuring for compatibility")
        
        # With OpenRouter, allow empty model name to use their default routing
        if not llm or llm.strip() == "":
            logger.info("Empty model name detected with OpenRouter - using OpenRouter's default model selection")
            # Use a default OpenAI model for best compatibility
            llm = "openai/gpt-3.5-turbo"
            logger.info(f"Using default model for OpenRouter: {llm}")
        
        # Ensure proper model name formatting for non-empty model names
        elif '/' not in llm:
            # For models without provider specification, assume OpenAI to be most compatible
            llm = f"openai/{llm}"
            logger.info(f"Modified model name for OpenRouter: {llm}")
        
        # Warning for non-OpenAI models - they might have compatibility issues
        if not llm.startswith("openai/"):
            logger.warning(f"Using non-OpenAI model with OpenRouter: {llm}")
            logger.warning("If you experience errors, try using an OpenAI model like 'openai/gpt-3.5-turbo'")
        
        # Create custom OpenAIModel for OpenRouter
        from pydantic_ai.models.openai import OpenAIModel as BaseOpenAIModel
        
        class OpenRouterModel(BaseOpenAIModel):
            """OpenRouter-compatible model that adds the necessary headers to each request."""
            
            def __init__(self, model_name, base_url=None, api_key=None):
                super().__init__(model_name, base_url=base_url, api_key=api_key)
                # Store base URL and API key for creating clients with different attribute names
                self._router_base_url = base_url
                self._router_api_key = api_key
            
            async def _completions_create(
                self,
                messages,
                stream,
                model_settings,
                model_request_parameters,
            ):
                """Override the _completions_create method to add OpenRouter headers."""
                from openai import OpenAI
                
                # Clean base URL to prevent double paths
                base = self._router_base_url.rstrip('/')
                if base.endswith('/api/v1'):
                    base = base  # Keep as is
                
                # Create a specific client for this request with the correct base URL
                client = OpenAI(
                    api_key=self._router_api_key,
                    base_url=base,
                )
                
                # Map messages just like the parent method does
                openai_messages = []
                for m in messages:
                    async for msg in self._map_message(m):
                        openai_messages.append(msg)
                
                # Add headers that OpenRouter requires
                headers = {
                    "HTTP-Referer": "https://pydantic-ai-mcp-agent.com",
                    "X-Title": "Pydantic AI MCP Agent"
                }
                
                # Use the standard kwargs from parent but add our headers
                kwargs = {
                    "model": self._model_name,
                    "messages": openai_messages,
                    "stream": stream,
                    "extra_headers": headers,  # This is the key change
                }
                
                # Add other optional parameters from model_settings
                for param, value in model_settings.items():
                    if value is not None and param != "openai_api_type" and param != "openai_organization":
                        kwargs[param] = value
                
                # Call the OpenAI client directly with our extra headers
                logger.info(f"Making OpenRouter request with model: {self._model_name}")
                
                try:
                    # Handle different response types from OpenRouter based on the model
                    response = client.chat.completions.create(**kwargs)
                    
                    # Create a wrapper class for the Stream that implements __aenter__ and __aexit__
                    class AsyncStreamWrapper:
                        """Wrapper to make a Stream object compatible with async context manager."""
                        def __init__(self, stream_obj):
                            self.stream = stream_obj
                            # Store the first chunk for error detection
                            self._first_chunk = None
                            # Track if we've started processing
                            self._started = False
                            # Cache for chunks
                            self._chunks = []
                            
                        async def __aenter__(self):
                            """Implement async context manager entry."""
                            return self
                            
                        async def __aexit__(self, exc_type, exc_value, traceback):
                            """Implement async context manager exit."""
                            pass
                        
                        def _extract_error_message(self, chunk):
                            """Extract error message from a chunk if present."""
                            try:
                                # Check for different error patterns
                                if hasattr(chunk, 'error') and chunk.error:
                                    return f"OpenRouter error: {chunk.error}"
                                
                                # Sometimes errors are nested in choices
                                if hasattr(chunk, 'choices') and chunk.choices:
                                    choice = chunk.choices[0]
                                    if hasattr(choice, 'finish_reason') and choice.finish_reason == 'content_filter':
                                        return "Content filtered by provider"
                                    if hasattr(choice, 'error') and choice.error:
                                        return f"Provider error: {choice.error}"
                                    
                                # Check for error in raw response
                                if hasattr(chunk, 'raw') and 'error' in getattr(chunk, 'raw', {}):
                                    return f"API error: {chunk.raw['error']}"
                                    
                                return None
                            except Exception as e:
                                logger.error(f"Error extracting error message: {e}")
                                return None
                        
                        def _get_chunks(self):
                            """Get chunks from synchronous iterator if not already processed."""
                            if not self._started:
                                try:
                                    # Try to get the first chunk to check for errors
                                    stream_iter = iter(self.stream)
                                    try:
                                        self._first_chunk = next(stream_iter)
                                        # Check if the first chunk contains an error
                                        error_msg = self._extract_error_message(self._first_chunk)
                                        if error_msg:
                                            logger.error(f"Detected error in stream: {error_msg}")
                                            # Return empty list and let the caller handle it
                                            self._chunks = []
                                            # Raise exception to break the processing
                                            raise ValueError(error_msg)
                                        
                                        # If no error, add first chunk and continue
                                        self._chunks = [self._first_chunk]
                                        # Add remaining chunks
                                        for chunk in stream_iter:
                                            self._chunks.append(chunk)
                                    except StopIteration:
                                        # No chunks available
                                        logger.warning("Stream iterator is empty")
                                        self._chunks = []
                                    
                                    logger.debug(f"Received {len(self._chunks)} chunks from stream")
                                except Exception as e:
                                    error_msg = str(e)
                                    if "Provider returned error" in error_msg:
                                        # This is a common OpenRouter error
                                        logger.error(f"OpenRouter error from provider: {error_msg}")
                                        # Re-raise with more descriptive message
                                        raise ValueError(f"The provider (Gemini) returned an error. Try an OpenAI model instead.")
                                    else:
                                        logger.error(f"Error converting stream to list: {e}")
                                    self._chunks = []
                                    # Propagate the error
                                    raise
                                finally:
                                    self._started = True
                            return self._chunks
                        
                        async def __aiter__(self):
                            """Make this an async iterator by yielding collected chunks."""
                            try:
                                chunks = self._get_chunks()
                                for chunk in chunks:
                                    yield chunk
                            except Exception as e:
                                # Re-raise the error to be caught by the caller
                                raise ValueError(f"Stream error: {str(e)}")
                        
                        async def stream_text(self, *, delta=False):
                            """Stream text from the completion incrementally."""
                            try:
                                # Collect chunks if needed
                                chunks = self._get_chunks()
                                
                                # Create artificial completion if there are no valid chunks
                                if not chunks:
                                    # Yield a message explaining the issue to avoid an empty response
                                    yield "I'm unable to process your request via this model. Please try with an OpenAI model like 'openai/gpt-3.5-turbo' instead."
                                    return
                                
                                for chunk in chunks:
                                    # Extract text based on mode (delta or full)
                                    text = None
                                    
                                    # Try to extract text from the chunk
                                    try:
                                        if delta:
                                            # Try to get delta.content first
                                            if hasattr(chunk, 'choices') and chunk.choices:
                                                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                                                    text = chunk.choices[0].delta.content
                                                # Fallback to message.content
                                                elif hasattr(chunk.choices[0], 'message') and hasattr(chunk.choices[0].message, 'content'):
                                                    text = chunk.choices[0].message.content
                                        else:
                                            # Get full message content
                                            if hasattr(chunk, 'choices') and chunk.choices:
                                                if hasattr(chunk.choices[0], 'message') and hasattr(chunk.choices[0].message, 'content'):
                                                    text = chunk.choices[0].message.content
                                    except Exception as e:
                                        logger.error(f"Error extracting text from chunk: {e}")
                                        continue
                                        
                                    # Yield the text if it exists
                                    if text is not None:
                                        yield text
                                        
                            except ValueError as e:
                                # Handle known errors with readable messages
                                if "Provider returned error" in str(e):
                                    yield "I'm unable to process your request with this model. Please try an OpenAI model like 'openai/gpt-3.5-turbo' instead."
                                else:
                                    yield f"Error processing response: {str(e)}"
                            except Exception as e:
                                logger.error(f"Error in stream_text: {e}")
                                yield "An error occurred while processing your request. Please try a different model."
                        
                        # Proxy all other attributes to the wrapped stream
                        def __getattr__(self, name):
                            return getattr(self.stream, name)
                    
                    # If the response is a Stream object and not awaitable, 
                    # we need to wrap it to make it compatible with async context managers
                    if hasattr(response, '__await__'):
                        # It's awaitable, we can await it
                        return await response
                    else:
                        # It's already a Stream object, wrap it to support async context manager
                        logger.info("OpenRouter returned a Stream object - wrapping for compatibility")
                        return AsyncStreamWrapper(response)
                        
                except Exception as e:
                    logger.error(f"Error creating OpenRouter completion: {e}", exc_info=True)
                    raise
        
        return OpenRouterModel(llm, base_url=base_url, api_key=api_key)
    
    # For non-OpenRouter, use the standard model
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
                    
                    # Special handling for OpenRouter errors
                    if 'openrouter' in os.getenv('BASE_URL', '').lower():
                        if 'Insufficient credits' in error_message:
                            error_message = "OpenRouter error: Insufficient credits. Please add more credits in your OpenRouter account: https://openrouter.ai/settings/credits"
                        elif 'No endpoints found matching your data policy' in error_message:
                            error_message = "OpenRouter error: Data policy restriction. Please enable prompt training in your OpenRouter settings: https://openrouter.ai/settings/privacy"
                        elif 'Provider returned error' in error_message:
                            advice = ("This error often occurs with non-OpenAI models on OpenRouter. "
                                     "Try using an OpenAI model like 'openai/gpt-3.5-turbo' instead, "
                                     "or leave the model field empty to use OpenRouter's default selection.")
                            error_message = f"OpenRouter error: {error_message}\n\nSuggestion: {advice}"
                    
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
            "model_choice": os.getenv('MODEL_CHOICE', 'gpt-4o-mini')
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