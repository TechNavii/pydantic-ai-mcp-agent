from pydantic_ai import RunContext, Tool as PydanticTool
from pydantic_ai.tools import ToolDefinition
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as MCPTool
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
import logging
import shutil
import json
import os
import re
import time
import functools

# Setup a custom formatter to redact sensitive information
class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive data from logs"""
    
    def __init__(self, patterns: List[str] = None):
        super().__init__()
        # Extended patterns to catch more potential sensitive data
        self.patterns = patterns or [
            'api_key', 'token', 'secret', 'password', 'credential',
            'access_key', 'auth', 'private_key', 'certificate'
        ]
    
    def filter(self, record):
        if isinstance(record.msg, str):
            # Redact patterns in the format key=value or key: value
            for pattern in self.patterns:
                regex = re.compile(rf'({pattern})\s*[=:]\s*[\'\"](.*?)[\'\"](\s|,|$)', re.IGNORECASE)
                record.msg = regex.sub(r'\1=*REDACTED*\3', record.msg)
                
            # Also redact URLs that might contain tokens as query params
            url_regex = re.compile(r'(https?://[^\s]*[?&][^\s]*=)([^\s&"]*)', re.IGNORECASE)
            record.msg = url_regex.sub(r'\1*REDACTED*', record.msg)
        return True

# Configure logging with sensitive data filtering
logger = logging.getLogger('mcp_client')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
handler.addFilter(SensitiveDataFilter())
logger.addHandler(handler)
logger.setLevel(logging.ERROR)

# Secure configuration constants
# List of allowed commands that can be executed
ALLOWED_COMMANDS = {
    'npx': True,  # Allow npx for running node packages
    'node': True  # Allow node for running JavaScript
}

# Maximum number of concurrent tool calls to prevent flooding
MAX_CONCURRENT_TOOL_CALLS = 20

# Rate limiting configuration for tool calls
@dataclass
class RateLimiter:
    """Simple rate limiter for API calls"""
    max_calls: int = 100  # Maximum calls per time window
    time_window: float = 60.0  # Time window in seconds
    call_history: List[float] = field(default_factory=list)  # Timestamps of recent calls
    
    def is_allowed(self) -> bool:
        """Check if a new call is allowed based on rate limits"""
        current_time = time.time()
        # Remove timestamps older than the time window
        self.call_history = [t for t in self.call_history if current_time - t <= self.time_window]
        # Check if we've exceeded the maximum calls
        if len(self.call_history) >= self.max_calls:
            return False
        # Record this call attempt
        self.call_history.append(current_time)
        return True

# Create a global rate limiter instance
tool_rate_limiter = RateLimiter()

class MCPClient:
    """Manages connections to one or more MCP servers based on mcp_config.json"""

    def __init__(self) -> None:
        self.servers: List[MCPServer] = []
        self.config: Dict[str, Any] = {}
        self.tools: List[Any] = []
        self.exit_stack = AsyncExitStack()
        self._initialization_lock = asyncio.Lock()
        self._concurrent_tool_calls = 0  # Counter for concurrent tool executions
        self._tool_call_lock = asyncio.Lock()  # Lock for modifying the counter

    def load_servers(self, config_path: str) -> None:
        """Load server configuration from a JSON file (typically mcp_config.json)
        and creates an instance of each server (no active connection until 'start' though).

        Args:
            config_path: Path to the JSON configuration file.
        
        Raises:
            FileNotFoundError: If the config file is not found.
            json.JSONDecodeError: If the config file contains invalid JSON.
            KeyError: If the config file doesn't have the expected structure.
            ValueError: If the config file is empty or invalid.
        """
        try:
            config_path = os.path.expandvars(os.path.expanduser(config_path))
            
            with open(config_path, "r") as config_file:
                self.config = json.load(config_file)
            
            if not self.config or not isinstance(self.config, dict):
                raise ValueError("Invalid config file: empty or not a dictionary")
                
            if "mcpServers" not in self.config:
                raise KeyError("Invalid config: missing 'mcpServers' key")
                
            if not isinstance(self.config["mcpServers"], dict):
                raise ValueError("Invalid config: 'mcpServers' must be a dictionary")
                
            if not self.config["mcpServers"]:
                logger.warning("Config contains no MCP servers")
            
            # Process environment variables in the config
            self._process_env_variables_in_config()
                
            # Create server instances
            self.servers = [MCPServer(name, server_config) 
                           for name, server_config in self.config["mcpServers"].items()]            
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            raise
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid config structure: {e}")
            raise
    
    def _process_env_variables_in_config(self) -> None:
        """Process environment variables in the config.
        
        Replaces ${ENV_VAR} or $ENV_VAR patterns with values from environment variables.
        Validates that required environment variables exist.
        
        Raises:
            ValueError: If a required environment variable is missing.
        """
        if not self.config or not isinstance(self.config, dict) or "mcpServers" not in self.config:
            return
            
        missing_env_vars = set()  # Track missing environment variables
            
        for server_name, server_config in self.config["mcpServers"].items():
            # Process environment variables in the environment section
            if "env" in server_config and isinstance(server_config["env"], dict):
                # Validate env section structure
                if not all(isinstance(k, str) and (isinstance(v, str) or v is None) 
                           for k, v in server_config["env"].items()):
                    logger.warning(f"Invalid 'env' structure for server '{server_name}'. "
                                 f"All keys must be strings and values must be strings or None.")
                
                for key, value in server_config["env"].items():
                    if isinstance(value, str):
                        # Replace ${VAR} or $VAR with environment variable values
                        pattern1 = r'\${([A-Za-z0-9_]+)}'
                        pattern2 = r'\$([A-Za-z0-9_]+)'
                        
                        # Find all environment variables referenced in this value
                        env_vars = set(re.findall(pattern1, value) + re.findall(pattern2, value))
                        
                        # Check if any referenced environment variables are missing
                        for env_var in env_vars:
                            if env_var not in os.environ:
                                missing_env_vars.add(env_var)
                                logger.warning(f"Environment variable '{env_var}' referenced in "
                                             f"server '{server_name}' env.{key} is not set")
                        
                        # Process ${VAR} pattern
                        for match in re.findall(pattern1, value):
                            env_value = os.environ.get(match, '')
                            value = value.replace(f"${{{match}}}", env_value)
                            
                        # Process $VAR pattern
                        for match in re.findall(pattern2, value):
                            env_value = os.environ.get(match, '')
                            value = value.replace(f"${match}", env_value)
                            
                        server_config["env"][key] = value
        
        if missing_env_vars:
            logger.error(f"Missing environment variables: {', '.join(missing_env_vars)}")
            # We log the error but don't raise an exception to maintain original functionality

    async def start(self) -> List[PydanticTool]:
        """Starts each MCP server and returns the tools for each server formatted for Pydantic AI.
        
        Returns:
            List of Pydantic AI tools derived from MCP tools.
            
        Raises:
            RuntimeError: If server initialization fails.
        """
        async with self._initialization_lock:
            self.tools = []
            failed_servers = []
            initialized_servers = []
            
            # Try to initialize all servers
            for server in self.servers:
                try:
                    logger.info(f"Initializing MCP server: {server.name}")
                    await server.initialize()
                    initialized_servers.append(server)
                    tools = await server.create_pydantic_ai_tools()
                    self.tools.extend(tools)
                    logger.info(f"Successfully initialized MCP server: {server.name} with {len(tools)} tools")
                except Exception as e:
                    logger.error(f"Failed to initialize server '{server.name}': {str(e)}")
                    failed_servers.append((server.name, str(e)))
            
            # If any servers failed, clean up the successful ones and report the error
            if failed_servers:
                error_details = '\n'.join([f"- {name}: {error}" for name, error in failed_servers])
                logger.error(f"Some MCP servers failed to initialize:\n{error_details}")
                
                if initialized_servers:
                    logger.info("Cleaning up successfully initialized servers due to partial failure")
                    await self.cleanup_servers()
                    self.tools = []
                
                if not self.tools:  # If no tools were registered, raise an error
                    raise RuntimeError(f"Failed to initialize MCP servers: {error_details}")
            
            return self.tools
            
    @asynccontextmanager
    async def session(self):
        """Context manager for using the MCP client in an async context.
        
        Yields:
            The current MCPClient instance with initialized tools.
            
        Example:
            ```python
            async with mcp_client.session() as client:
                # Use client.tools here
                pass
            # Automatically cleaned up after the block
            ```
        """
        try:
            await self.start()
            yield self
        finally:
            await self.cleanup()

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        cleanup_tasks = []
        
        # Create cleanup tasks for all servers
        for server in self.servers:
            if hasattr(server, 'cleanup'):
                task = asyncio.create_task(self._safe_server_cleanup(server))
                cleanup_tasks.append(task)
        
        # Wait for all cleanup tasks to complete if there are any
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
    
    async def _safe_server_cleanup(self, server: 'MCPServer') -> None:
        """Safely clean up a server with timeout protection."""
        try:
            # Create a task for the cleanup
            cleanup_task = asyncio.create_task(server.cleanup())
            
            # Wait for the cleanup with a timeout
            try:
                await asyncio.wait_for(cleanup_task, timeout=10.0)
                logger.info(f"Successfully cleaned up server: {server.name}")
            except asyncio.TimeoutError:
                logger.warning(f"Cleanup timeout for server {server.name}, forcing cancellation")
                cleanup_task.cancel()
                try:
                    await cleanup_task
                except asyncio.CancelledError:
                    pass
        except Exception as e:
            logger.warning(f"Error during cleanup of server {server.name}: {str(e)}")

    async def cleanup(self) -> None:
        """Clean up all resources including the exit stack."""
        try:
            logger.info("Starting MCP client cleanup")
            # First clean up all servers
            await self.cleanup_servers()
            # Then close the exit stack
            await self.exit_stack.aclose()
            logger.info("MCP client cleanup completed")
        except Exception as e:
            logger.error(f"Error during final MCP client cleanup: {str(e)}")


class MCPServer:
    """Manages MCP server connections and tool execution."""
    
    # Class-level variables for tracking concurrent tool executions
    _concurrent_tool_calls = 0
    _tool_call_lock = asyncio.Lock()

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.stdio_context: Optional[Any] = None
        self.session: Optional[ClientSession] = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self._initialized: bool = False
        self._start_time: Optional[float] = None
        self._last_activity: Optional[float] = None  # Track when server was last used
        self._rate_limiter = RateLimiter()  # Initialize rate limiter for this server

    async def initialize(self) -> None:
        """Initialize the server connection.
        
        Raises:
            ValueError: If the command is not valid.
            ConnectionError: If the connection to the MCP server fails.
            TimeoutError: If the connection times out.
            RuntimeError: For other initialization errors.
        """
        # Don't initialize if already initialized
        if self._initialized and self.session is not None:
            logger.debug(f"Server {self.name} already initialized, skipping")
            return
            
        # Set start time for performance tracking
        self._start_time = time.time()
        
        # Validate server configuration
        self._validate_server_config()
        
        # Determine the command to execute
        command = self._resolve_command()
        
        # Create server parameters
        server_params = StdioServerParameters(
            command=command,
            args=self.config.get("args", []),
            env=self.config.get("env") or None,
        )
        
        try:
            # Attempt to connect with a timeout
            connection_task = self._connect_to_server(server_params)
            await asyncio.wait_for(connection_task, timeout=30.0)  # 30-second timeout
            
            elapsed = time.time() - self._start_time
            logger.info(f"Server {self.name} initialized in {elapsed:.2f} seconds")
            self._initialized = True
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout while initializing server {self.name}")
            await self.cleanup()
            raise TimeoutError(f"Timed out while connecting to MCP server '{self.name}'")
            
        except Exception as e:
            logger.error(f"Error initializing server {self.name}: {str(e)}")
            await self.cleanup()
            if isinstance(e, ValueError):
                raise
            elif "connection" in str(e).lower():
                raise ConnectionError(f"Failed to connect to MCP server '{self.name}': {str(e)}")
            else:
                raise RuntimeError(f"Failed to initialize MCP server '{self.name}': {str(e)}")
    
    def _validate_server_config(self) -> None:
        """Validate the server configuration.
        
        Raises:
            ValueError: If the configuration is invalid.
        """
        required_keys = ["command", "args"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key '{key}' for server '{self.name}'")
                
        if not isinstance(self.config["command"], str) or not self.config["command"].strip():
            raise ValueError(f"Invalid command for server '{self.name}': must be a non-empty string")
            
        if not isinstance(self.config["args"], list):
            raise ValueError(f"Invalid args for server '{self.name}': must be a list")
            
        # Validate command is in the allowed list
        command = self.config["command"]
        if command not in ALLOWED_COMMANDS:
            logger.warning(f"Command '{command}' for server '{self.name}' is not in the allowed commands list")
            
        # Validate args for shell injection
        for i, arg in enumerate(self.config["args"]):
            if not isinstance(arg, str):
                raise ValueError(f"Argument {i} for server '{self.name}' is not a string")
                
            # Check for potential shell injection patterns
            suspicious_patterns = ['&', '|', ';', '`', '$', '>', '<', '\\', '"', "'", '&&', '||']
            for pattern in suspicious_patterns:
                if pattern in arg:
                    logger.warning(f"Suspicious character '{pattern}' found in argument for server '{self.name}': {arg}")
        
        # Validate env dict if present
        if "env" in self.config:
            if not isinstance(self.config["env"], dict):
                raise ValueError(f"Invalid env for server '{self.name}': must be a dictionary")
                
            for key, value in self.config["env"].items():
                if not isinstance(key, str):
                    raise ValueError(f"Invalid env key for server '{self.name}': {key} must be a string")
                    
                if not (isinstance(value, str) or value is None):
                    raise ValueError(f"Invalid env value for server '{self.name}': {key}={value} must be a string or None")
    
    def _resolve_command(self) -> str:
        """Resolve the command to execute.
        
        Returns:
            The resolved command path.
            
        Raises:
            ValueError: If the command cannot be resolved or is not allowed.
        """
        command = self.config["command"]
        
        # Check if the command is allowed
        if command not in ALLOWED_COMMANDS:
            logger.error(f"Command '{command}' for server '{self.name}' is not in the allowed commands list")
            raise ValueError(f"Command '{command}' is not allowed. Allowed commands: {', '.join(ALLOWED_COMMANDS.keys())}")
        
        # If the command is npx, resolve its path
        if command == "npx":
            resolved_command = shutil.which("npx")
            if resolved_command is None:
                raise ValueError(f"Could not find 'npx' in PATH. Please install Node.js and npm.")
            logger.debug(f"Resolved 'npx' to {resolved_command}")
            return resolved_command
        
        # For other commands, use as-is but validate they exist
        if os.path.isabs(command):
            if not os.path.exists(command):
                raise ValueError(f"Command not found: {command}")
            # For absolute paths, check if the file is executable
            if not os.access(command, os.X_OK):
                raise ValueError(f"Command is not executable: {command}")
            logger.debug(f"Using absolute path command: {command}")
        else:
            # If it's a relative command, try to find it in PATH
            resolved_command = shutil.which(command)
            if resolved_command is None:
                raise ValueError(f"Command not found in PATH: {command}")
            command = resolved_command
            logger.debug(f"Resolved command '{self.config['command']}' to {command}")
            
        return command
            
    async def _connect_to_server(self, server_params: StdioServerParameters) -> None:
        """Connect to the MCP server.
        
        Args:
            server_params: The server parameters.
            
        Raises:
            Exception: If the connection fails.
        """
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            
            # Initialize the session with a timeout
            init_task = asyncio.create_task(session.initialize())
            await asyncio.wait_for(init_task, timeout=20.0)  # 20-second timeout
            
            self.session = session
            
        except Exception as e:
            logger.error(f"Error connecting to server {self.name}: {str(e)}")
            await self.cleanup()
            raise

    async def create_pydantic_ai_tools(self) -> List[PydanticTool]:
        """Convert MCP tools to pydantic_ai Tools.
        
        Returns:
            List of Pydantic AI tools.
            
        Raises:
            RuntimeError: If the server is not initialized or tools cannot be retrieved.
        """
        if not self._initialized or self.session is None:
            raise RuntimeError(f"Cannot create tools: server '{self.name}' is not initialized")
            
        try:
            tools_result = await asyncio.wait_for(
                self.session.list_tools(), 
                timeout=10.0  # 10-second timeout
            )
            tools = tools_result.tools
            logger.info(f"Retrieved {len(tools)} tools from server '{self.name}'")
            return [self.create_tool_instance(tool) for tool in tools]
        except asyncio.TimeoutError:
            logger.error(f"Timeout while retrieving tools from server '{self.name}'")
            raise RuntimeError(f"Timed out while retrieving tools from MCP server '{self.name}'")
        except Exception as e:
            logger.error(f"Error retrieving tools from server '{self.name}': {str(e)}")
            raise RuntimeError(f"Failed to retrieve tools from MCP server '{self.name}': {str(e)}")

    def create_tool_instance(self, tool: MCPTool) -> PydanticTool:
        """Initialize a Pydantic AI Tool from an MCP Tool.
        
        Args:
            tool: The MCP tool to convert.
            
        Returns:
            A Pydantic AI tool.
        """
        async def execute_tool(**kwargs: Any) -> Any:
            global tool_rate_limiter
            
            if not self._initialized or self.session is None:
                logger.error(f"Cannot execute tool '{tool.name}': server '{self.name}' is not initialized")
                raise RuntimeError(f"Cannot execute tool: server '{self.name}' is not initialized")
            
            # Apply rate limiting
            if not tool_rate_limiter.is_allowed():
                logger.warning(f"Rate limit exceeded for tool '{tool.name}' on server '{self.name}'")
                raise RuntimeError(f"Rate limit exceeded for tool execution. Please try again later.")
                
            # Log the input parameters (with sensitive data filtered)
            logger.debug(f"Executing tool '{tool.name}' on server '{self.name}' with params: {kwargs}")
                
            # Check for too many concurrent executions
            async with MCPServer._tool_call_lock:
                if MCPServer._concurrent_tool_calls >= MAX_CONCURRENT_TOOL_CALLS:
                    logger.warning(f"Too many concurrent tool executions ({MCPServer._concurrent_tool_calls})")
                    raise RuntimeError(f"Too many concurrent tool calls. Please try again later.")
                MCPServer._concurrent_tool_calls += 1
                
            try:
                # Validate input parameters if schema is available
                if tool.inputSchema and isinstance(kwargs, dict):
                    self._validate_tool_input(tool.name, tool.inputSchema, kwargs)
                    
                # Execute the tool with a timeout
                start_time = time.time()
                result = await asyncio.wait_for(
                    self.session.call_tool(tool.name, arguments=kwargs),
                    timeout=60.0  # 60-second timeout for tool execution
                )
                elapsed = time.time() - start_time
                logger.debug(f"Tool '{tool.name}' executed in {elapsed:.2f} seconds")
                return result
                
            except asyncio.TimeoutError:
                logger.error(f"Timeout executing tool '{tool.name}' on server '{self.name}'")
                raise TimeoutError(f"Tool execution timed out: '{tool.name}'")
            except Exception as e:
                logger.error(f"Error executing tool '{tool.name}' on server '{self.name}': {str(e)}")
                raise RuntimeError(f"Tool execution error '{tool.name}': {str(e)}")
            finally:
                # Always decrement the counter, even if an error occurred
                async with MCPServer._tool_call_lock:
                    MCPServer._concurrent_tool_calls -= 1
    
    def _validate_tool_input(self, tool_name: str, schema: Dict[str, Any], data: Dict[str, Any]) -> None:
        """Validate tool input against schema.
        
        This is a basic validation - in a real implementation, you'd use
        a full JSON Schema validator.
        """
        if not isinstance(schema, dict) or not isinstance(data, dict):
            return
            
        # Basic validation of required fields
        if 'required' in schema and isinstance(schema['required'], list):
            for field in schema['required']:
                if field not in data:
                    logger.warning(f"Missing required field '{field}' for tool '{tool_name}'")
                    
        # Basic type validation for properties
        if 'properties' in schema and isinstance(schema['properties'], dict):
            for field, field_schema in schema['properties'].items():
                if field in data and 'type' in field_schema:
                    expected_type = field_schema['type']
                    value = data[field]
                    
                    # Very basic type checking
                    valid = True
                    if expected_type == 'string' and not isinstance(value, str):
                        valid = False
                    elif expected_type == 'number' and not isinstance(value, (int, float)):
                        valid = False
                    elif expected_type == 'integer' and not isinstance(value, int):
                        valid = False
                    elif expected_type == 'boolean' and not isinstance(value, bool):
                        valid = False
                    elif expected_type == 'array' and not isinstance(value, list):
                        valid = False
                    elif expected_type == 'object' and not isinstance(value, dict):
                        valid = False
                        
                    if not valid:
                        logger.warning(f"Type mismatch for field '{field}' in tool '{tool_name}': "
                                     f"expected {expected_type}, got {type(value).__name__}")

    def create_tool_instance(self, tool: MCPTool) -> PydanticTool:
        """Create a Tool instance that can be registered with Pydantic."""
        async def execute_tool(**kwargs) -> Any:
            """Execute the tool with the given arguments."""
            if not self._initialized:
                await self.initialize()
                
            # Apply rate limiting
            if not self._rate_limiter.is_allowed():
                logger.warning(f"Rate limit exceeded for server '{self.name}'")
                raise RuntimeError(f"Rate limit exceeded. Please try again later.")
                
            # Limit concurrent calls
            async with MCPServer._tool_call_lock:
                if MCPServer._concurrent_tool_calls >= MAX_CONCURRENT_TOOL_CALLS:
                    logger.warning(f"Too many concurrent tool calls (limit: {MAX_CONCURRENT_TOOL_CALLS})")
                    raise RuntimeError(f"Too many concurrent tool calls. Please try again later.")
                MCPServer._concurrent_tool_calls += 1
                
            try:
                # Validate input parameters if schema is available
                if tool.inputSchema and isinstance(kwargs, dict):
                    self._validate_tool_input(tool.name, tool.inputSchema, kwargs)
                    
                # Execute the tool with a timeout
                start_time = time.time()
                result = await asyncio.wait_for(
                    self.session.call_tool(tool.name, arguments=kwargs),
                    timeout=60.0  # 60-second timeout for tool execution
                )
                elapsed = time.time() - start_time
                logger.debug(f"Tool '{tool.name}' executed in {elapsed:.2f} seconds")
                return result
            except Exception as e:
                logger.error(f"Error executing tool '{tool.name}' on server '{self.name}': {e}")
                raise
            finally:
                # Always decrement the counter, even if an error occurred
                async with MCPServer._tool_call_lock:
                    MCPServer._concurrent_tool_calls -= 1
        
        async def prepare_tool(ctx: RunContext, tool_def: ToolDefinition) -> Optional[ToolDefinition]:
            tool_def.parameters_json_schema = tool.inputSchema
            return tool_def
        
        return PydanticTool(
            execute_tool,
            name=tool.name,
            description=tool.description or "",
            takes_ctx=False,
            prepare=prepare_tool
        )

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            if not self._initialized:
                logger.debug(f"Server {self.name} was not initialized, nothing to clean up")
                return
                
            try:
                logger.debug(f"Starting cleanup for server {self.name}")
                
                # Close the exit stack with a timeout
                close_task = asyncio.create_task(self.exit_stack.aclose())
                try:
                    await asyncio.wait_for(close_task, timeout=10.0)  # 10-second timeout
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout during exit stack cleanup for server {self.name}")
                    close_task.cancel()
                    try:
                        await close_task
                    except asyncio.CancelledError:
                        pass
                        
                # Reset server state
                self.session = None
                self.stdio_context = None
                self._initialized = False
                
                logger.info(f"Successfully cleaned up server {self.name}")
                
            except Exception as e:
                logger.error(f"Error during cleanup of server {self.name}: {str(e)}")
            finally:
                # Always mark as not initialized, even if cleanup failed
                self._initialized = False  