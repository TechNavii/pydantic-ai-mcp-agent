from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import asyncio
import json
import os
import pathlib
import logging
from dotenv import load_dotenv, set_key, find_dotenv

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai import exceptions  # Add the exceptions import
import mcp_client
from pydantic_ai import messages as pydantic_messages

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get the directory where the current script is located
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
# Define the path to the config file relative to the script directory
CONFIG_FILE = SCRIPT_DIR / "mcp_config.json"
# Define the static files directory
STATIC_DIR = SCRIPT_DIR / "static"

# --- Define OpenRouterModel and AsyncStreamWrapper at the top level ---
from pydantic_ai.models.openai import OpenAIModel as BaseOpenAIModel
from openai import OpenAI # Keep OpenAI import needed for OpenRouterModel

class AsyncStreamWrapper:
    """Wrapper to make a synchronous Stream object compatible with async context manager."""
    def __init__(self, stream_obj):
        self.stream = stream_obj
        self._first_chunk = None
        self._started = False
        self._chunks = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        pass

    def _extract_error_message(self, chunk):
         try:
            if hasattr(chunk, 'error') and chunk.error: return f"OpenRouter error: {chunk.error}"
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, 'finish_reason') and choice.finish_reason == 'content_filter': return "Content filtered by provider"
                if hasattr(choice, 'error') and choice.error: return f"Provider error: {choice.error}"
            if hasattr(chunk, 'raw') and 'error' in getattr(chunk, 'raw', {}): return f"API error: {chunk.raw['error']}"
            return None
         except Exception as e: logger.error(f"[AsyncStreamWrapper] Error extracting error message: {e}"); return None

    def _get_chunks(self):
        # This runs synchronously to exhaust the iterator initially
        if not self._started:
            try:
                stream_iter = iter(self.stream)
                try:
                    self._first_chunk = next(stream_iter)
                    error_msg = self._extract_error_message(self._first_chunk)
                    if error_msg:
                        logger.error(f"[AsyncStreamWrapper] Detected error in stream: {error_msg}")
                        self._chunks = []
                        raise ValueError(error_msg) # Raise error to be caught by __aiter__ or stream_text
                    self._chunks = [self._first_chunk]
                    # Consume the rest of the synchronous stream
                    for chunk in stream_iter: self._chunks.append(chunk)
                except StopIteration: 
                    logger.warning("[AsyncStreamWrapper] Stream iterator was empty on first next() call.")
                    self._chunks = [] 
                    # If first chunk had error, it would have been raised already
                    # If no first chunk, it's just empty, not necessarily an error yet
                except ValueError as ve:
                     raise ve # Propagate error detected in first chunk
                except Exception as iter_exc:
                     logger.error(f"[AsyncStreamWrapper] Error during initial stream iteration: {iter_exc}", exc_info=True)
                     self._chunks = []
                     # Re-raise as ValueError for consistent handling
                     raise ValueError(f"Error processing stream: {iter_exc}")
                     
                logger.debug(f"[AsyncStreamWrapper] Received {len(self._chunks)} chunks from stream")
            finally: 
                self._started = True
        return self._chunks

    async def __aiter__(self):
        # This makes the class async iterable
        try:
            # Ensure chunks are loaded synchronously first
            chunks = self._get_chunks() 
            for chunk in chunks:
                yield chunk
        except ValueError as ve:
             # Catch errors raised during _get_chunks (like first chunk error)
             logger.error(f"[AsyncStreamWrapper] Error during async iteration setup: {ve}")
             # Special handling for OpenRouter provider errors detected in _get_chunks
             if "Provider returned error" in str(ve):
                  # Make error more specific for OpenRouter context
                  raise ValueError(f"OpenRouter Error: The underlying provider ({self.stream.model if hasattr(self.stream, 'model') else 'unknown'}) returned an error. This model may be incompatible or unavailable via OpenRouter. Try an OpenAI model instead.")
             else:
                 # Re-raise other ValueErrors caught during chunk loading
                 raise ValueError(f"Stream error: {str(ve)}")
        except Exception as e:
            logger.error(f"[AsyncStreamWrapper] Unexpected error during async iteration: {e}", exc_info=True)
            raise ValueError(f"Unexpected stream error: {str(e)}")

    async def stream_text(self, *, delta=False):
        # Stream text content, handling potential errors during chunk loading
        try:
            # Ensure chunks are loaded, catching potential errors
            chunks = self._get_chunks() 
            if not chunks:
                logger.warning("[AsyncStreamWrapper] No chunks available to stream text.")
                # Check if an error occurred during loading (should have been raised by _get_chunks)
                # Yield a message indicating potential loading issue
                yield "[Info: Stream empty or error occurred during initialization]"
                return

            for chunk in chunks:
                text = None
                try:
                    if delta:
                        if hasattr(chunk, 'choices') and chunk.choices:
                            delta_data = chunk.choices[0].delta
                            if hasattr(delta_data, 'content'): text = delta_data.content
                    else: # Non-delta
                         if hasattr(chunk, 'choices') and chunk.choices:
                             message_data = chunk.choices[0].message
                             if hasattr(message_data, 'content'): text = message_data.content
                except Exception as e: logger.error(f"[AsyncStreamWrapper] Error extracting text from chunk: {e}"); continue
                if text is not None: yield text
        except ValueError as ve:
             # Catch errors raised during _get_chunks
             logger.error(f"[AsyncStreamWrapper] Error during stream_text setup: {ve}")
             # Provide specific feedback based on the error
             if "Provider returned error" in str(ve):
                 # Use the refined error message
                 yield f"[OpenRouter Error: The underlying provider returned an error. This model may be incompatible or unavailable via OpenRouter. Try an OpenAI model instead.]"
             elif "Stream error:" in str(ve):
                 yield f"[{str(ve)}]" # Pass specific stream errors through
             else:
                  yield f"[Error processing stream response: {str(ve)}]"
        except Exception as e:
            logger.error(f"[AsyncStreamWrapper] Unexpected error in stream_text: {e}", exc_info=True)
            yield "[An unexpected error occurred while processing the stream.]"

    def __getattr__(self, name):
        # Proxy other attributes to the wrapped stream object if needed
        return getattr(self.stream, name)

class OpenRouterModel(BaseOpenAIModel):
    """OpenRouter-compatible model that adds required headers and handles stream wrapping."""
    def __init__(self, model_name, base_url=None, api_key=None):
        super().__init__(model_name, base_url=base_url, api_key=api_key)
        # Store base URL and API key separately for clarity
        self._router_base_url = base_url
        self._router_api_key = api_key

    async def _completions_create(
        self,
        messages,
        stream,
        model_settings,
        model_request_parameters,
    ):
        """Override _completions_create to add headers and wrap stream."""
        base = self._router_base_url.rstrip('/')
        # Ensure /v1 endpoint for OpenRouter compatibility
        if not base.endswith('/v1'):
             if base.endswith('/api'): # Handle cases like '.../api'
                 base = f"{base}/v1"
             elif not base.endswith('/'): # Ensure trailing slash before adding /v1 if needed
                 base = f"{base}/v1"
             else:
                  base = f"{base}v1"
             logger.info(f"Adjusted OpenRouter base URL to: {base}")


        client = OpenAI(api_key=self._router_api_key, base_url=base)

        openai_messages = []
        for m in messages:
            async for msg in self._map_message(m):
                openai_messages.append(msg)

        headers = {
            "HTTP-Referer": "https://pydantic-ai-mcp-agent.com", # Replace if needed
            "X-Title": "Pydantic AI MCP Agent"
        }

        kwargs = {
            "model": self._model_name,
            "messages": openai_messages,
            "stream": stream,
            "extra_headers": headers,
        }

        for param, value in model_settings.items():
            if value is not None and param not in ["openai_api_type", "openai_organization"]:
                 kwargs[param] = value

        logger.info(f"Making OpenRouter request with model: {self._model_name}")

        try:
            response = client.chat.completions.create(**kwargs)

            # If streaming, wrap the synchronous stream from openai client v1+
            if stream and not hasattr(response, '__aiter__'):
                logger.info("OpenRouter returned a non-async stream - wrapping for compatibility")
                return AsyncStreamWrapper(response)
            else:
                # If already async (shouldn't happen with current openai lib?) or not streaming
                return response
        except Exception as e:
            # Catch API errors during the create call itself
            logger.error(f"Error creating OpenRouter completion: {e}", exc_info=True)
            # Check for common OpenRouter errors here if possible, e.g., 402 Payment Required
            if hasattr(e, 'status_code'):
                 if e.status_code == 402:
                     raise exceptions.ModelError("OpenRouter error: Insufficient credits. Please add credits at https://openrouter.ai/settings/credits")
                 elif e.status_code == 429: # Rate limit
                     raise exceptions.ModelError("OpenRouter error: Rate limit hit. Please check your limits or wait.")
            # Re-raise generic error if not specifically handled
            raise exceptions.ModelError(f"OpenRouter API request failed: {e}")
# --- End of top-level definitions ---

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

def get_model() -> Union[OpenAIModel, AnthropicModel, GeminiModel, OpenRouterModel]: # Add OpenRouterModel to hint
    """Get the configured model based on provider and settings."""
    llm = os.getenv('MODEL_CHOICE', 'gpt-4o-mini')
    base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
    api_key = os.getenv('LLM_API_KEY', 'no-api-key-provided')

    logger.info(f"Attempting to configure model: {llm} with base URL: {base_url}")

    # Check for Google Gemini API
    if 'generativelanguage.googleapis.com' in base_url.lower():
        logger.info("Google Gemini API detected - using built-in Gemini model")
        try:
            model_name = llm
            if not model_name.startswith('models/'):
                 model_name = f"models/{model_name}"
                 logger.info(f"Prepended 'models/' to Gemini model name: {model_name}")
            return GeminiModel(model_name, api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize built-in Gemini Model: {e}", exc_info=True)
            raise ValueError(f"Failed to configure Gemini. Check API key and model name ('{llm}'). Error: {e}")

    # Check for Anthropic API
    elif 'api.anthropic.com' in base_url.lower():
        logger.info("Anthropic API detected - using built-in Anthropic model")
        try:
            # Instantiate the built-in AnthropicModel
            # Pass the original model name directly from env var (llm)
            # Removed the logic that stripped the date suffix
            model_name = llm 
            logger.info(f"Using Anthropic model name as provided: {model_name}")
            return AnthropicModel(model_name, api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize built-in Anthropic Model: {e}", exc_info=True)
            raise ValueError(f"Failed to configure Anthropic. Check API key and model name ('{llm}'). Error: {e}")
    
    # Check for DeepSeek API (using standard OpenAIModel)
    elif 'api.deepseek.com' in base_url.lower():
        logger.info("DeepSeek API detected - using standard OpenAIModel")
        return OpenAIModel(
            llm,
            base_url=base_url, # Use DeepSeek's base URL
            api_key=api_key
        )
        
    # Check if using OpenRouter
    elif 'openrouter' in base_url.lower():
        # --- Revert to using the top-level OpenRouterModel class ---
        logger.info("OpenRouter detected - configuring custom OpenRouterModel for compatibility")
        
        openrouter_llm = llm 
        # Apply model name formatting specific to OpenRouter recommendations
        if not openrouter_llm or openrouter_llm.strip() == "":
            openrouter_llm = "openai/gpt-3.5-turbo" # Default suggested by OpenRouter
            logger.info(f"Using default model for OpenRouter: {openrouter_llm}")
        elif '/' not in openrouter_llm:
             openrouter_llm = f"openai/{openrouter_llm}"
             logger.info(f"Assuming openai prefix for OpenRouter model: {openrouter_llm}")

        # Warning for non-OpenAI models (still relevant)
        if not openrouter_llm.startswith("openai/"):
            logger.warning(f"Using non-OpenAI model ({openrouter_llm}) with OpenRouter. Tool calling and streaming might be unreliable.")

        # Instantiate the custom model defined at the top level
        # This uses the AsyncStreamWrapper for better streaming compatibility
        return OpenRouterModel(openrouter_llm, base_url=base_url, api_key=api_key)
        # --- End Reverted OpenRouter Handling ---
    
    # Default to standard OpenAI model
    else:
        if not base_url or not base_url.startswith('http'):
             logger.warning(f"Invalid or empty BASE_URL provided: '{base_url}'. Falling back to default OpenAI URL.")
             base_url = 'https://api.openai.com/v1' # Default OpenAI URL

        logger.info(f"Defaulting to standard OpenAI model for base URL: {base_url}")
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
    active_model_instance = None # Store the instantiated model
    
    try:
        # Modify agent initialization to store the model instance
        logger.info("Starting agent initialization...")
        client = mcp_client.MCPClient()
        client.load_servers(str(CONFIG_FILE))
        tools = await client.start()
        active_model_instance = get_model() # Get and store the model
        mcp_agent = Agent(model=active_model_instance, tools=tools)
        mcp_agent_client = client # Assign client after successful start
        logger.info("MCP client and Pydantic AI agent initialized successfully.")
        
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

                    # ===== Execute Agent using agent.run() =====
                    logger.info("Starting agent.run()...")
                    agent_run_result: Optional[Any] = None 
                    final_text_response = ""
                    final_tool_names_found = set()
                    agent_history = []

                    try:
                        agent_run_result = await mcp_agent.run(
                            current_msg_content, 
                            message_history=[] # Start fresh each time
                        )
                        logger.info("Agent run completed.")
                        
                        # --- Detailed Inspection of AgentRunResult --- 
                        if agent_run_result:
                            logger.info(f"Agent run returned object of type: {type(agent_run_result)}")
                            
                            # --- Try extracting text ---
                            possible_text_attrs = ['output', 'content', 'response']
                            for attr in possible_text_attrs:
                                if hasattr(agent_run_result, attr):
                                    value = getattr(agent_run_result, attr)
                                    # Check if it's a callable method
                                    if callable(value):
                                         try:
                                             value = value() # Call the method
                                             logger.debug(f"  Called method result.{attr}()")
                                         except Exception as call_err:
                                              logger.warning(f"  Failed to call result.{attr}(): {call_err}")
                                              value = None
                                              
                                    logger.debug(f"  Checking text attribute '{attr}', resolved type: {type(value)}")
                                    if isinstance(value, str) and value.strip():
                                        final_text_response = value.strip()
                                        logger.info(f"  Extracted final text from result.{attr}: '{final_text_response[:100]}...'")
                                        break
                            if not final_text_response and isinstance(agent_run_result, str) and agent_run_result.strip():
                                final_text_response = agent_run_result.strip()
                                logger.info(f"  Agent run returned string directly: '{final_text_response[:100]}...'")

                            # --- Try extracting history ---
                            history_attr_found = None
                            possible_history_attrs = ['all_messages', 'messages', 'history'] # Prioritize all_messages
                            for attr in possible_history_attrs:
                                if hasattr(agent_run_result, attr):
                                     value = getattr(agent_run_result, attr)
                                     # Check if it's a callable method
                                     if callable(value):
                                         try:
                                             value = value() # Call the method
                                             logger.debug(f"  Called method result.{attr}()")
                                         except Exception as call_err:
                                              logger.warning(f"  Failed to call result.{attr}(): {call_err}")
                                              value = None
                                              
                                     logger.debug(f"  Checking history attribute '{attr}', resolved type: {type(value)}")
                                     if isinstance(value, list):
                                         agent_history = value
                                         history_attr_found = attr
                                         logger.info(f"  Extracted history from result.{attr} ({len(agent_history)} messages).")
                                         break

                            if not history_attr_found:
                                logger.warning("Could not extract message history list from AgentRunResult.")

                            # --- Inspect History Extracted from Result (if found) --- 
                            if agent_history:
                                logger.debug(f"--- Inspecting History Extracted from Result (using {history_attr_found}) --- ")
                                current_run_tool_names = set()
                                last_text_part_content = "" # Variable to store the latest text part found
                                
                                for i, msg in enumerate(agent_history):
                                    msg_type = type(msg).__name__
                                    msg_role = getattr(msg, 'role', 'N/A')
                                    logger.debug(f"  History Msg {i}: Type={msg_type}, Role={msg_role}")

                                    if isinstance(msg, pydantic_messages.ModelResponse):
                                        if hasattr(msg, 'parts') and isinstance(msg.parts, list):
                                            for part_idx, part in enumerate(msg.parts):
                                                part_type = type(part).__name__
                                                # logger.debug(f"    Part {part_idx}: Type={part_type}") # Verbose
                                                
                                                # Find ToolCallParts to identify used tools
                                                if isinstance(part, pydantic_messages.ToolCallPart):
                                                    tool_name = getattr(part, 'tool_name', 'unknown_tool')
                                                    logger.info(f"    ToolCallPart Found: Name={tool_name}")
                                                    current_run_tool_names.add(tool_name) 
                                                
                                                # Find TextPart and store its content, overwriting previous ones
                                                elif isinstance(part, pydantic_messages.TextPart):
                                                    if hasattr(part, 'content') and isinstance(part.content, str) and part.content.strip():
                                                         current_text = part.content.strip()
                                                         logger.debug(f"    Found TextPart content: '{current_text[:100]}...'")
                                                         last_text_part_content = current_text # Always store the latest
                                    
                                # Assign tools found in this history
                                final_tool_names_found = current_run_tool_names 
                                
                                # Use the last found text part content ONLY if direct extraction failed
                                if not final_text_response and last_text_part_content:
                                    final_text_response = last_text_part_content
                                    logger.info(f"Using text from the *last* TextPart found in history: '{final_text_response[:100]}...'")
                                    
                                logger.debug(f"--- Finished Inspecting History from Result (found tools: {final_tool_names_found}) --- ")
                            else:
                                # If history couldn't be extracted, log it
                                logger.warning("No history list was extracted, cannot check for tools or fallback text.")

                        else:
                            logger.warning("Agent run returned None or empty result.")

                    # ===== Correct position for the except block catching agent run errors =====
                    except Exception as agent_run_error:
                         logger.error(f"Error during agent run or result processing: {agent_run_error}", exc_info=True)
                         await websocket.send_text(json.dumps({"type": "error", "content": f"Error during agent processing: {str(agent_run_error)}"}))
                         continue # Skip to next websocket message cycle
                    
                    # ===== Send TOOL Notifications =====
                    if final_tool_names_found:
                        logger.info(f"Sending tool usage info: {final_tool_names_found}")
                        tool_list = sorted(list(final_tool_names_found)) 
                        for tool_name in tool_list:
                             logger.info(f"Sending tool_used message for {tool_name}.")
                             await websocket.send_text(json.dumps({"type": "tool_used", "tool_name": tool_name}))

                    # ===== Send COMPLETION Message =====
                    final_text_to_send = final_text_response 

                    if final_text_to_send:
                        logger.debug(f"Sending final completion text length: {len(final_text_to_send)}")
                        await websocket.send_text(json.dumps({"type": "complete", "content": final_text_to_send}))
                        logger.info("Complete message sent.")
                    elif final_tool_names_found and not final_text_to_send:
                         logger.info("Tools used, but no final text response could be extracted. Sending generic confirmation.")
                         await websocket.send_text(json.dumps({"type": "complete", "content": "[Tool(s) used successfully.]"})) 
                    elif not final_tool_names_found and not final_text_to_send:
                        logger.warning("Agent run produced no text response AND no tool calls detected.")
                        await websocket.send_text(json.dumps({"type": "error", "content": "Agent produced no response or tool calls."}))

                # ===== Outer error handling for JSON/Parsing errors (ensure correct indentation) =====
                except json.JSONDecodeError as json_error:
                    logger.error(f"JSON decode error: {json_error}", exc_info=True)
                    await websocket.send_text(json.dumps({"type": "error","content": "Invalid message format received"}))
                    continue # Continue websocket loop
                except Exception as outer_processing_error: # Catch any other unexpected errors
                    logger.error(f"Unexpected error in message processing loop: {outer_processing_error}", exc_info=True)
                    await websocket.send_text(json.dumps({"type": "error","content": f"Unexpected server error: {str(outer_processing_error)}"}))
                    continue # Continue websocket loop
            
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON decode error: {json_error}", exc_info=True)
                await websocket.send_text(json.dumps({"type": "error","content": "Invalid message format received"}))
            except Exception as parse_error: # Catch any other errors during message processing
                logger.error(f"Message processing error: {parse_error}", exc_info=True)
                await websocket.send_text(json.dumps({"type": "error","content": f"Failed to process message: {str(parse_error)}"}))
                
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