---
title: MCP Implementation Guide
sidebar_position: 3
description: Practical guide to implementing Model Context Protocol in applications
---

# Implementing Model Context Protocol

This guide provides practical approaches for implementing Model Context Protocol (MCP) in your applications, with code examples and best practices.

## Getting Started with MCP

To implement MCP in your application, you'll need to:

1. Choose an implementation approach (direct API, abstraction library, or custom)
2. Set up the messaging structure
3. Configure function calling capabilities
4. Implement context management

## Basic Implementation Pattern

Here's a minimal MCP implementation using direct API integration with OpenAI:

```python
import openai
import json
import os

# Configure API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

class BasicMCPClient:
    def __init__(self, model_name="gpt-4"):
        self.model_name = model_name
        self.messages = []
        
    def add_system_message(self, content):
        """Add a system message to guide model behavior"""
        self.messages.append({"role": "system", "content": content})
        
    def add_user_message(self, content):
        """Add a user message to the conversation"""
        self.messages.append({"role": "user", "content": content})
        
    def get_response(self, functions=None):
        """Get a response from the model using MCP"""
        kwargs = {
            "model": self.model_name,
            "messages": self.messages,
        }
        
        # Add functions if provided
        if functions:
            kwargs["functions"] = functions
            kwargs["function_call"] = "auto"
            
        # Get completion from the model
        response = openai.ChatCompletion.create(**kwargs)
        
        # Process the response
        message = response.choices[0].message
        self.messages.append({"role": "assistant", "content": message.content})
        
        # Handle function calls if present
        if hasattr(message, "function_call") and message.function_call:
            return {
                "content": message.content,
                "function_call": {
                    "name": message.function_call.name,
                    "arguments": json.loads(message.function_call.arguments)
                }
            }
        
        return {"content": message.content}
    
    def add_function_response(self, function_name, content):
        """Add a function response to the conversation"""
        self.messages.append({
            "role": "function", 
            "name": function_name, 
            "content": json.dumps(content) if not isinstance(content, str) else content
        })
```

## Using the Basic MCP Implementation

```python
# Example usage
client = BasicMCPClient(model_name="gpt-4")

# Initialize with system instructions
client.add_system_message("You are a helpful assistant with access to tools.")

# Define a function
get_weather_function = {
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name with optional state/country"
            }
        },
        "required": ["location"]
    }
}

# Add user query
client.add_user_message("What's the weather like in Seattle?")

# Get response with function capability
response = client.get_response(functions=[get_weather_function])

# Check if function call is requested
if "function_call" in response:
    fc = response["function_call"]
    print(f"Function called: {fc['name']} with args: {fc['arguments']}")
    
    # Simulate getting weather data
    weather_data = {
        "temperature": 65,
        "condition": "Partly cloudy",
        "humidity": 72,
        "location": "Seattle, WA"
    }
    
    # Add function response
    client.add_function_response(fc["name"], weather_data)
    
    # Get final response
    final_response = client.get_response()
    print(f"Final response: {final_response['content']}")
else:
    print(f"Response: {response['content']}")
```

## Implementing with LangChain

LangChain provides a higher-level abstraction for working with MCP:

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage, FunctionMessage
from langchain.tools import StructuredTool
from typing import Optional

# Define function for getting weather
def get_weather(location: str) -> dict:
    """Get current weather for a location"""
    # In a real implementation, call a weather service
    return {
        "temperature": 65,
        "condition": "Partly cloudy", 
        "humidity": 72,
        "location": location
    }

# Create a structured tool
weather_tool = StructuredTool.from_function(
    name="get_weather",
    func=get_weather,
    description="Get current weather for a location",
    return_direct=False
)

# Initialize the chat model with tools
model = ChatOpenAI(
    temperature=0,
    model_name="gpt-4",
    streaming=True
).bind_tools([weather_tool])

# Create conversation messages
messages = [
    SystemMessage(content="You are a helpful assistant with access to tools."),
    HumanMessage(content="What's the weather like in Seattle?")
]

# Get response
response = model.invoke(messages)

# Process response
print(f"Response: {response.content}")

# Add messages to conversation
messages.append(response)

# Continue the conversation if needed
messages.append(HumanMessage(content="How does that compare to New York?"))
response2 = model.invoke(messages)
print(f"Follow-up response: {response2.content}")
```

## Advanced Implementation Patterns

### Streaming with MCP

```python
def stream_with_mcp(client, user_message, functions=None):
    """Demonstrate streaming with MCP"""
    client.add_user_message(user_message)
    
    # Set up streaming
    kwargs = {
        "model": client.model_name,
        "messages": client.messages,
        "stream": True
    }
    
    if functions:
        kwargs["functions"] = functions
        kwargs["function_call"] = "auto"
    
    # Get streaming response
    response_stream = openai.ChatCompletion.create(**kwargs)
    
    # Collect content in parts
    collected_content = ""
    function_call_parts = []
    
    for chunk in response_stream:
        delta = chunk.choices[0].delta
        
        # Handle content streaming
        if hasattr(delta, "content") and delta.content:
            collected_content += delta.content
            print(delta.content, end="", flush=True)
        
        # Handle function call streaming
        if hasattr(delta, "function_call"):
            if hasattr(delta.function_call, "name") and delta.function_call.name:
                function_call_parts.append(("name", delta.function_call.name))
            if hasattr(delta.function_call, "arguments") and delta.function_call.arguments:
                function_call_parts.append(("arguments", delta.function_call.arguments))
    
    print("\n")  # End the streaming output with a newline
    
    # Update the client messages with the complete response
    client.messages.append({"role": "assistant", "content": collected_content})
    
    # Process function call if present
    if function_call_parts:
        name = ""
        args_str = ""
        
        for part_type, content in function_call_parts:
            if part_type == "name":
                name += content
            elif part_type == "arguments":
                args_str += content
        
        if name and args_str:
            # Add function call to response
            return {
                "content": collected_content,
                "function_call": {
                    "name": name,
                    "arguments": json.loads(args_str)
                }
            }
    
    return {"content": collected_content}
```

### Context Window Management

```python
def manage_context_window(messages, max_tokens=4000, model="gpt-4"):
    """Manage the context window by pruning old messages if needed"""
    from tiktoken import encoding_for_model
    
    # Initialize the tokenizer
    enc = encoding_for_model(model)
    
    # Calculate current token count
    token_count = 0
    for msg in messages:
        # Count tokens in content
        token_count += len(enc.encode(msg.get("content", "")))
        
        # Count tokens in function name and content if present
        if msg.get("name"):
            token_count += len(enc.encode(msg["name"]))
        
        # Add overhead for each message (role, formatting)
        token_count += 4
    
    # If we're under the limit, return the original messages
    if token_count <= max_tokens:
        return messages
    
    # Otherwise, prune messages while preserving system and recent messages
    system_messages = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]
    
    # Keep removing old non-system messages until under token limit
    while token_count > max_tokens and non_system:
        # Remove oldest non-system message
        removed_msg = non_system.pop(0)
        # Subtract its tokens
        token_count -= len(enc.encode(removed_msg.get("content", "")))
        if removed_msg.get("name"):
            token_count -= len(enc.encode(removed_msg["name"]))
        token_count -= 4  # Message overhead
    
    # Recombine messages
    pruned_messages = system_messages + non_system
    
    return pruned_messages
```

### Error Handling with Retries

```python
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def mcp_call_with_retry(client, user_message, functions=None):
    """Make an MCP call with retry logic for handling temporary errors"""
    try:
        client.add_user_message(user_message)
        return client.get_response(functions=functions)
    except openai.error.RateLimitError:
        # Handle rate limiting
        time.sleep(5)  # Wait before retry
        raise
    except openai.error.APIError as e:
        # Handle API errors
        if e.http_status >= 500:  # Server errors
            raise  # Let tenacity retry
        else:
            # Client errors shouldn't be retried
            return {"error": str(e), "content": "An error occurred processing your request."}
```

## Handling Conversation State

For applications that need to persist conversation state:

```python
import pickle

class PersistentMCPClient(BasicMCPClient):
    """An MCP client that can save and load conversation state"""
    
    def save_state(self, filename):
        """Save the conversation state to a file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.messages, f)
    
    def load_state(self, filename):
        """Load conversation state from a file"""
        with open(filename, 'rb') as f:
            self.messages = pickle.load(f)
    
    def export_conversation(self, format="text"):
        """Export the conversation in different formats"""
        if format == "text":
            result = []
            for msg in self.messages:
                if msg["role"] == "system":
                    result.append(f"[System] {msg['content']}")
                elif msg["role"] == "user":
                    result.append(f"[User] {msg['content']}")
                elif msg["role"] == "assistant":
                    result.append(f"[Assistant] {msg['content']}")
                elif msg["role"] == "function":
                    result.append(f"[Function: {msg['name']}] {msg['content']}")
            return "\n\n".join(result)
        elif format == "json":
            return json.dumps(self.messages, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
```

## Best Practices

When implementing MCP in your applications, consider these best practices:

1. **Clear System Instructions**: Start conversations with clear system messages
2. **Context Window Management**: Implement strategies to handle context limits
3. **Error Handling**: Always implement robust error handling
4. **Function Registry**: Maintain a registry of available functions with detailed schemas
5. **Progressive Enhancement**: Fall back gracefully when advanced features aren't available
6. **Security**: Validate function inputs and implement proper authorization
7. **Testing**: Create test suites for MCP interactions and function calling

## Next Steps

- Explore [MCP Tools & Frameworks](./tools-frameworks.md) for additional libraries and tools
- Review real-world [Case Studies](./case-studies.md) of MCP implementations
- Check out the [Core Concepts](./core-concepts.md) if you need a refresher on fundamental principles