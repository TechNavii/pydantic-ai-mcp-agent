class ChatApp {
    constructor() {
        this.ws = null;
        this.messageHistory = [];
        this.isTyping = false;
        this.currentAssistantMessage = '';
        this.currentAssistantMessageDiv = null;
        this.tools = [];
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;

        // DOM elements
        this.messagesContainer = document.getElementById('chat-messages');
        this.messageInput = document.getElementById('message-input');
        this.sendButton = document.getElementById('send-button');
        this.statusContainer = document.querySelector('.status-container');
        this.statusIndicator = this.statusContainer.querySelector('.status-indicator');
        this.statusText = this.statusContainer.querySelector('.status-text');
        this.toolsPanel = document.getElementById('tools-panel');
        this.toolsList = document.getElementById('tools-list');
        this.toggleToolsBtn = document.getElementById('toggle-tools-btn');
        this.closeToolsBtn = document.querySelector('.close-tools-btn');

        // Configuration elements
        this.configModal = document.getElementById('config-modal');
        this.configBtn = document.getElementById('config-btn');
        this.configForm = document.getElementById('config-form');
        this.closeConfigBtn = document.querySelector('.close-modal-btn');
        this.cancelConfigBtn = document.getElementById('cancel-config');
        this.togglePasswordBtn = document.querySelector('.toggle-password');

        // Templates
        this.messageTemplate = document.getElementById('message-template');
        this.loadingTemplate = document.getElementById('loading-template');
        this.errorTemplate = document.getElementById('error-template');
        this.toolTemplate = document.getElementById('tool-template');

        // Initialize
        this.initWebSocket();
        this.setupEventListeners();
        this.loadConfiguration();
        
        console.log('ChatApp initialized');
    }

    setupEventListeners() {
        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            const scrollHeight = this.messageInput.scrollHeight;
            const maxHeight = 150;
            this.messageInput.style.height = `${Math.min(scrollHeight, maxHeight)}px`;
        });

        // Send button click
        this.sendButton.addEventListener('click', () => {
            console.log('Send button clicked');
            this.sendMessage();
        });

        // Keydown listener for Enter
        this.messageInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                console.log('Enter key pressed');
                this.sendMessage();
            }
        });

        // Initially disable input controls
        this.messageInput.disabled = true;
        this.sendButton.disabled = true;

        // Tools panel toggle
        this.toggleToolsBtn.addEventListener('click', () => {
            console.log('Toggle tools button clicked');
            this.toggleTools();
        });

        this.closeToolsBtn.addEventListener('click', () => {
            console.log('Close tools button clicked');
            this.toggleTools();
        });

        // Configuration modal
        this.configBtn.addEventListener('click', () => this.showConfigModal());
        this.closeConfigBtn.addEventListener('click', () => this.hideConfigModal());
        this.cancelConfigBtn.addEventListener('click', () => this.hideConfigModal());
        this.configForm.addEventListener('submit', (e) => this.handleConfigSubmit(e));
        this.togglePasswordBtn.addEventListener('click', () => this.togglePasswordVisibility());
    }

    initWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/chat`;
        console.log(`Connecting to WebSocket: ${wsUrl}`);

        if (this.ws) {
            this.ws.close();
        }

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connection established');
            this.messageInput.disabled = false;
            this.sendButton.disabled = false;
            this.updateConnectionStatus(true);
            this.addSystemMessage('Connected to the chat server.');
        };

        this.ws.onclose = (event) => {
            console.log('WebSocket connection closed:', event.code, event.reason);
            this.messageInput.disabled = true;
            this.sendButton.disabled = true;
            this.updateConnectionStatus(false);
            this.addSystemMessage('Connection lost. Attempting to reconnect...');
            setTimeout(() => this.initWebSocket(), 5000);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.messageInput.disabled = true;
            this.sendButton.disabled = true;
            this.updateConnectionStatus(false, 'Error');
            this.addSystemMessage('WebSocket connection error.');
        };

        this.ws.onmessage = (event) => {
            console.debug('Raw WebSocket message received:', event.data);
            try {
                const data = JSON.parse(event.data);
                console.log('[WebSocket Handler] Parsed message:', data);
                if (!data || typeof data !== 'object' || !data.type) {
                    throw new Error('Invalid message format');
                }

                switch (data.type) {
                    case 'tools':
                        if (Array.isArray(data.content)) {
                            this.handleToolsUpdate(data.content);
                        } else {
                            console.warn('Invalid tools data format:', data.content);
                        }
                        break;
                    case 'delta':
                        if (typeof data.content === 'string') {
                            this.updateAssistantMessage(data.content);
                        } else {
                            console.warn('Invalid delta data format:', data.content);
                        }
                        break;
                    case 'complete':
                        console.log("[WebSocket Handler] Received 'complete' message. Content:", data.content);
                        if (typeof data.content === 'string') {
                            this.finalizeAssistantMessage(data.content);
                        } else {
                            console.warn('Invalid complete data format:', data.content);
                        }
                        break;
                    case 'tool_used':
                        console.log("[WebSocket Handler] Received 'tool_used' message:", data);
                        if (data.tool_name && typeof data.tool_name === 'string') {
                            // Simplify display for testing
                            this.addSystemMessage(`✅ Tool used: ${data.tool_name}`); 
                            // this.displayToolUsedMessage(data.tool_name); // Keep original commented out
                        } else {
                            console.warn('Invalid tool_used data format:', data);
                        }
                        break;
                    case 'error':
                        this.showError(data.content || 'Unknown error occurred');
                        if (this.isTyping) {
                            this.finalizeAssistantMessage('', true);
                        }
                        break;
                    default:
                        console.warn('Received unknown message type:', data.type);
                }
            } catch (e) {
                console.error('Failed to parse WebSocket message:', e, '\nRaw message:', event.data);
                this.showError('Error processing server response');
                if (this.isTyping) {
                    this.finalizeAssistantMessage('', true);
                }
            }
        };
    }

    handleToolsUpdate(tools) {
        console.log('Handling tools update:', tools);
        if (!Array.isArray(tools)) {
            console.error('Invalid tools data:', tools);
            return;
        }
        this.tools = tools;
        try {
            this.renderTools();
        } catch (error) {
            console.error('Error rendering tools:', error);
        }
    }

    renderTools() {
        if (!this.toolsList || !this.toolTemplate) {
            console.error('Required DOM elements not found');
            return;
        }

        try {
            this.toolsList.innerHTML = '';
            
            this.tools.forEach((tool, index) => {
                try {
                    const toolElement = this.toolTemplate.content.cloneNode(true);
                    const toolDiv = toolElement.querySelector('.tool');
                    if (!toolDiv) {
                        console.error('Tool template structure not found');
                        return;
                    }

                    // Add tool ID and data attributes
                    toolDiv.setAttribute('data-tool-id', index);
                    toolDiv.setAttribute('data-tool-name', tool.name || '');

                    // Set name and description safely
                    const nameElement = toolDiv.querySelector('.tool-name');
                    const descElement = toolDiv.querySelector('.tool-description');
                    const paramsElement = toolDiv.querySelector('.tool-parameters');

                    if (nameElement) {
                        nameElement.textContent = tool.name || 'Unnamed Tool';
                    }
                    if (descElement) {
                        descElement.textContent = tool.description || 'No description available';
                    }
                    if (paramsElement && tool.parameters) {
                        try {
                            const formattedParams = JSON.stringify(tool.parameters, null, 2);
                            paramsElement.textContent = formattedParams;
                        } catch (jsonError) {
                            console.warn('Error formatting tool parameters:', jsonError);
                            paramsElement.textContent = 'Parameters not available';
                        }
                    }

                    // Add click handler for expansion
                    const toolHeader = toolDiv.querySelector('.tool-name');
                    if (toolHeader) {
                        toolHeader.addEventListener('click', () => {
                            toolDiv.classList.toggle('expanded');
                        });
                    }

                    this.toolsList.appendChild(toolDiv);
                } catch (toolError) {
                    console.error('Error rendering individual tool:', toolError);
                }
            });

            console.log('Tools rendered successfully');
        } catch (error) {
            console.error('Error in renderTools:', error);
        }
    }

    toggleTools() {
        if (!this.toolsPanel) {
            console.error('Tools panel element not found');
            return;
        }

        try {
            console.log('Toggling tools panel');
            const isVisible = this.toolsPanel.classList.toggle('visible');
            this.toggleToolsBtn.setAttribute('aria-expanded', isVisible.toString());
            console.log('Tools panel visibility:', isVisible);
        } catch (error) {
            console.error('Error toggling tools panel:', error);
        }
    }

    updateConnectionStatus(isConnected, status = null) {
        if (isConnected) {
            this.statusIndicator.style.backgroundColor = 'var(--success-color)';
            this.statusText.textContent = 'Connected';
        } else {
            this.statusIndicator.style.backgroundColor = 'var(--error-color)';
            this.statusText.textContent = status || 'Disconnected';
        }
    }

    addMessage(content, role, isTyping = false) {
        const messageDiv = this.messageTemplate.content.cloneNode(true).querySelector('.message');
        messageDiv.classList.add(role);
        if (isTyping) messageDiv.classList.add('typing');
        
        const messageContent = messageDiv.querySelector('.message-content');
        if (content) {
            messageContent.innerHTML = this.formatMarkdown(content);
        }
        
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
        return messageDiv;
    }

    addSystemMessage(content) {
        console.log('[addSystemMessage] Called with:', content);
        const messageDiv = this.messageTemplate.content.cloneNode(true).querySelector('.message');
        messageDiv.classList.add('system');
        messageDiv.querySelector('.message-content').textContent = content;
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    updateAssistantMessage(delta) {
        if (!this.isTyping) {
            this.isTyping = true;
            this.currentAssistantMessage = '';
            this.currentAssistantMessageDiv = this.addMessage('', 'assistant', true);
        }

        try {
            this.currentAssistantMessage += delta;
            const messageContent = this.currentAssistantMessageDiv.querySelector('.message-content');
            if (messageContent) {
                messageContent.innerHTML = this.formatMarkdown(this.currentAssistantMessage);
                this.scrollToBottom();
            }
        } catch (error) {
            console.error('Error updating assistant message:', error);
        }
    }

    finalizeAssistantMessage(finalContent, isError = false) {
        console.log(`[finalizeAssistantMessage] Called. isTyping: ${this.isTyping}, finalContent: "${finalContent ? finalContent.substring(0, 50) + '...' : ''}", isError: ${isError}`);

        let targetDiv = this.currentAssistantMessageDiv;

        // If we weren't typing (no deltas received) but have final content, create a div now.
        if (!this.isTyping && finalContent && !isError) {
            console.log('[finalizeAssistantMessage] No prior deltas, creating new message div for final content.');
            targetDiv = this.addMessage('', 'assistant', false); // Create non-typing div
            this.currentAssistantMessageDiv = targetDiv; // Assign it so the rest of the logic can use it
        }
        
        // If there's no target div after the above check (e.g., error occurred before typing started, or empty final content),
        // we still need to proceed to the finally block to re-enable input.
        if (!targetDiv) {
             console.warn('[finalizeAssistantMessage] No targetDiv found. Proceeding to finally block.');
        } else { 
            // Only attempt to update content if we have a target div
            try {
                this.isTyping = false; // Ensure typing state is reset
                const messageContent = targetDiv.querySelector('.message-content');
                if (messageContent) {
                    const content = finalContent || this.currentAssistantMessage; // Use finalContent if available
                    console.log('[finalizeAssistantMessage] Setting final HTML content.');
                    messageContent.innerHTML = this.formatMarkdown(content);
                    targetDiv.classList.remove('typing'); // Remove typing indicator

                    if (!isError && typeof content === 'string' && content.trim() !== '') { // Only add non-empty messages to history
                        this.messageHistory.push({
                            role: 'assistant',
                            content: content
                        });
                        if (this.messageHistory.length > 50) {
                            this.messageHistory = this.messageHistory.slice(-50);
                        }
                        console.log('[finalizeAssistantMessage] Updated message history.');
                    }
                } else {
                     console.warn('[finalizeAssistantMessage] messageContent element not found within targetDiv.');
                }
            } catch (error) {
                console.error('Error finalizing assistant message content:', error);
                // Ensure we still proceed to finally block
            }
        }
        
        // --- Finally block MUST run to re-enable input --- 
        console.log('[finalizeAssistantMessage] Entering finally block.');
        this.currentAssistantMessage = '';
        this.currentAssistantMessageDiv = null; // Clear the reference
        this.isTyping = false; // Ensure typing state is false
        this.sendButton.disabled = false;
        this.messageInput.disabled = false;
        console.log('[finalizeAssistantMessage] Input re-enabled.');
        this.messageInput.focus();
        this.scrollToBottom(); // Scroll after final content is set
    }

    showError(error) {
        console.error('Chat Error:', error);
        try {
            const errorDiv = this.errorTemplate.content.cloneNode(true).querySelector('.error-message');
            if (!errorDiv) {
                console.error('Error template not found');
                return;
            }

            const errorText = errorDiv.querySelector('.error-text');
            const errorMessage = typeof error === 'string' ? error : 'An unknown error occurred';
            
            if (errorText) {
                errorText.textContent = errorMessage;
            } else {
                errorDiv.textContent = errorMessage;
            }
            
            this.messagesContainer.appendChild(errorDiv);
            this.scrollToBottom();
        } catch (e) {
            console.error('Error showing error message:', e);
        }
    }

    formatMarkdown(content) {
        if (typeof content !== 'string') return '';
        
        // Configure marked options
        marked.setOptions({
            highlight: function(code, lang) {
                const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                return hljs.highlight(code, { language }).value;
            },
            langPrefix: 'hljs language-',
            breaks: true,
            gfm: true
        });

        try {
            return marked(content);
        } catch (e) {
            console.error('Markdown parsing error:', e);
            return content;
        }
    }

    sendMessage() {
        console.log('sendMessage called');
        const message = this.messageInput.value.trim();
        
        if (!message) {
            console.log('Message is empty');
            return;
        }

        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            console.warn('WebSocket not connected');
            this.addSystemMessage('Cannot send message: Not connected to server');
            return;
        }

        console.log('Sending message:', message);

        // Add user message to UI
        this.addMessage(message, 'user');
        
        // Send to server
        try {
            // Get last 10 messages from history
            const recentHistory = this.messageHistory.slice(-10);
            console.log('Recent message history:', recentHistory);
            
            this.ws.send(JSON.stringify({
                message: message,
                history: recentHistory
            }));
            
            // Add current message to history after sending
            this.messageHistory.push({
                role: 'user',
                content: message
            });
            
            // Trim history if it gets too long
            if (this.messageHistory.length > 50) {
                this.messageHistory = this.messageHistory.slice(-50);
            }
            
            console.log('Updated message history:', this.messageHistory);
        } catch (error) {
            console.error('Failed to send message:', error);
            this.showError('Failed to send message to server');
            this.messageInput.disabled = false;
            this.sendButton.disabled = false;
            return;
        }

        // Clear and disable input
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        this.messageInput.disabled = true;
        this.sendButton.disabled = true;
    }

    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    // Configuration methods
    async loadConfiguration() {
        try {
            const response = await fetch('/api/config');
            if (response.ok) {
                const config = await response.json();
                document.getElementById('base-url').value = config.base_url || '';
                document.getElementById('api-key').value = config.api_key || '';
                document.getElementById('model-choice').value = config.model_choice || 'gpt-4o-mini';
            }
        } catch (error) {
            console.error('Error loading configuration:', error);
        }
    }

    async handleConfigSubmit(event) {
        event.preventDefault();
        const formData = new FormData(this.configForm);
        const baseUrl = formData.get('base-url');
        
        const config = {
            base_url: baseUrl,
            api_key: formData.get('api-key'),
            model_choice: formData.get('model-choice') || ''
        };
        
        // Show a note when using OpenRouter with empty model
        if (baseUrl.includes('openrouter') && !config.model_choice) {
            this.addSystemMessage('Using OpenRouter with automatic model selection');
        }

        try {
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });

            if (response.ok) {
                this.hideConfigModal();
                this.showSuccess('Configuration updated successfully');
                // Reconnect WebSocket with new configuration
                if (this.ws) {
                    this.ws.close();
                }
            } else {
                throw new Error('Failed to update configuration');
            }
        } catch (error) {
            console.error('Error saving configuration:', error);
            this.showError('Failed to save configuration');
        }
    }

    showConfigModal() {
        this.configModal.classList.add('visible');
    }

    hideConfigModal() {
        this.configModal.classList.remove('visible');
    }

    togglePasswordVisibility() {
        const apiKeyInput = document.getElementById('api-key');
        const icon = this.togglePasswordBtn.querySelector('.material-icons');
        if (apiKeyInput.type === 'password') {
            apiKeyInput.type = 'text';
            icon.textContent = 'visibility';
        } else {
            apiKeyInput.type = 'password';
            icon.textContent = 'visibility_off';
        }
    }

    showSuccess(message) {
        // You can implement this to show a success message
        const successElement = this.errorTemplate.content.cloneNode(true);
        const successDiv = successElement.querySelector('.error-message');
        successDiv.classList.add('success');
        successDiv.querySelector('.material-icons').textContent = 'check_circle';
        successDiv.querySelector('.error-text').textContent = message;
        this.messagesContainer.appendChild(successElement);
        this.scrollToBottom();
    }

    displayToolUsedMessage(toolName) {
        console.log(`Displaying tool used message: ${toolName}`);
        const messageElement = this.messageTemplate.content.cloneNode(true);
        const messageDiv = messageElement.querySelector('.message');
        const contentDiv = messageDiv.querySelector('.message-content');
        const avatar = messageDiv.querySelector('.avatar');

        messageDiv.classList.add('system', 'tool-used-message'); // Style as system message
        avatar.remove(); // Remove avatar for system-like message
        
        contentDiv.innerHTML = `<i>✅ Tool used: ${this.escapeHtml(toolName)}</i>`; 

        const cursor = messageDiv.querySelector('.cursor');
        if (cursor) cursor.remove();

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }
}

// Initialize the chat app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing ChatApp');
    window.chatApp = new ChatApp();
});

// Global function for toggling tools panel
window.toggleTools = function() {
    console.log('Toggle tools called');
    if (window.chatApp) {
        window.chatApp.toggleTools();
    }
}
