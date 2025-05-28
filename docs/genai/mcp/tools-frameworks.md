---
title: MCP Tools & Frameworks
sidebar_position: 4
description: Popular tools and frameworks for Model Context Protocol development
---

# MCP Tools & Frameworks

This guide explores the ecosystem of tools, libraries, and frameworks that support Model Context Protocol (MCP) development.

## Provider SDKs

### OpenAI SDK

The OpenAI Python library provides low-level access to OpenAI's API with MCP support:

```python
import openai

# Configure the API key
openai.api_key = "your-api-key"

# Basic MCP message exchange
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the capital of France?"}
    ]
)

# Access the response
print(response.choices[0].message.content)
```

**Key Features**:
- Direct API access
- Full function calling support
- Streaming responses
- Comprehensive error handling

**Resources**:
- [OpenAI Python Library](https://github.com/openai/openai-python)
- [Chat Completions API Documentation](https://platform.openai.com/docs/guides/text-generation/chat-completions-api)

### Anthropic SDK

Anthropic's Claude SDK supports the core MCP message format:

```python
from anthropic import Anthropic

# Initialize client
anthropic = Anthropic(api_key="your-anthropic-api-key")

# Create a message with MCP structure
response = anthropic.messages.create(
    model="claude-3-opus-20240229",
    system="You are a helpful assistant.",
    messages=[
        {"role": "user", "content": "What is the capital of Japan?"}
    ]
)

# Access the response
print(response.content[0].text)
```

**Key Features**:
- Claude model access
- System prompt support
- Tool use capability (in newer versions)

**Resources**:
- [Anthropic API Documentation](https://docs.anthropic.com/claude/reference/messages_post)

## Abstraction Libraries

### LangChain

LangChain provides a high-level, unified interface for working with different LLM providers using MCP concepts:

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType

# Initialize a chat model
chat = ChatOpenAI(
    model_name="gpt-4",
    temperature=0
)

# Create messages in MCP format
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the distance from Earth to Mars?")
]

# Get a response
response = chat.invoke(messages)
print(response.content)
```

**Key Features**:
- Unified interface across providers
- Built-in tools and agents
- Memory systems
- Chain composition for complex workflows
- Document loading and processing

**Resources**:
- [LangChain Documentation](https://python.langchain.com/docs/get_started)
- [LangChain Chat Models](https://python.langchain.com/docs/modules/model_io/models/chat)

### LlamaIndex

LlamaIndex focuses on data ingestion and retrieval while supporting MCP for model interactions:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

# Load documents
documents = SimpleDirectoryReader('data').load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Create query engine with MCP-compatible model
llm = OpenAI(model="gpt-4")
query_engine = index.as_query_engine(llm=llm)

# Query with MCP under the hood
response = query_engine.query("What information is in these documents?")
print(response)
```

**Key Features**:
- Document ingestion and indexing
- Query engines for RAG
- Sub-question decomposition
- Tool integrations
- Evaluation frameworks

**Resources**:
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [LlamaIndex Modules](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/)

### Semantic Kernel

Microsoft's Semantic Kernel provides .NET and Python SDKs with MCP support:

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

# Create kernel
kernel = sk.Kernel()

# Add OpenAI chat completion service
kernel.add_chat_service(
    "chat", 
    OpenAIChatCompletion("gpt-4", "your-api-key")
)

# Create a semantic function using natural language
summarize = kernel.create_semantic_function(
    "Summarize the following text in 3 sentences: {{$input}}"
)

# Run the function with MCP under the hood
result = summarize("Long text to summarize...")
print(result)
```

**Key Features**:
- Plugin architecture
- Semantic functions
- Planning capabilities
- Connectors for multiple providers
- Memory integration

**Resources**:
- [Semantic Kernel Documentation](https://learn.microsoft.com/en-us/semantic-kernel/)
- [Python SDK](https://github.com/microsoft/semantic-kernel/tree/main/python)

### Haystack

Deepset's Haystack library combines RAG with LLM capabilities, supporting MCP formats:

```python
from haystack import Pipeline
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.dataclasses import Document

# Set up a retriever with documents
docs = [
    Document(content="Paris is the capital of France."),
    Document(content="Berlin is the capital of Germany.")
]
retriever = InMemoryBM25Retriever(documents=docs)

# Create an OpenAI generator with MCP support
generator = OpenAIGenerator(api_key="your-api-key", model="gpt-4")

# Connect in a pipeline
pipe = Pipeline()
pipe.add_component("retriever", retriever)
pipe.add_component("generator", generator)

# Connect components
pipe.connect("retriever.documents", "generator.documents")

# Run the pipeline with MCP under the hood
result = pipe.run({"retriever": {"query": "What is the capital of France?"}})
print(result["generator"]["replies"][0])
```

**Key Features**:
- Pipeline architecture
- Multiple retrievers
- Document stores
- Evaluation tools
- Custom component creation

**Resources**:
- [Haystack Documentation](https://haystack.deepset.ai/)
- [Haystack LLM Integration](https://haystack.deepset.ai/integrations/llm)

## Specialized MCP Tools

### Function-Calling Libraries

#### OpenAI Function Calling Toolkit

```python
# Example with OpenAI function toolkit
from openai_function_calling import OpenAISchema
from pydantic import Field
from typing import List

class WeatherRequest(OpenAISchema):
    """Get the current weather in a given location"""
    location: str = Field(..., description="City and state or country")
    unit: str = Field("celsius", description="Temperature unit (celsius or fahrenheit)")

def get_weather_data(location: str, unit: str = "celsius"):
    # Mock implementation
    return {"temperature": 22, "unit": unit, "condition": "Sunny", "location": location}

# Use with the OpenAI API
functions = [WeatherRequest.openai_schema]
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "What's the weather in Paris?"}
    ],
    functions=functions,
    function_call="auto"
)

# Process the function call
if response.choices[0].message.function_call:
    call = response.choices[0].message.function_call
    args = json.loads(call.arguments)
    
    # Type-checked function call
    weather_req = WeatherRequest(**args)
    result = get_weather_data(weather_req.location, weather_req.unit)
    
    # Continue the conversation with the result
    print(f"Weather in {result['location']}: {result['temperature']}°{result['unit'][0].upper()}, {result['condition']}")
```

#### Instructor

The Instructor library simplifies function calling with Pydantic models:

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

# Enable instructor with OpenAI client
client = instructor.patch(OpenAI())

class WeatherRequest(BaseModel):
    """Get the current weather in a given location"""
    location: str = Field(..., description="City and state or country")
    unit: str = Field("celsius", description="Temperature unit (celsius or fahrenheit)")

# Direct extraction using MCP and function calling
weather_req = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "What's the weather in Tokyo?"}
    ],
    response_model=WeatherRequest
)

print(f"Request: {weather_req.location}, {weather_req.unit}")
```

### Litellm

LiteLLM provides a unified interface for multiple LLM providers with MCP support:

```python
from litellm import completion

# Works with OpenAI-compatible completion
response = completion(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello world"}
    ]
)

print(response["choices"][0]["message"]["content"])

# Easily switch to different providers
azure_response = completion(
    model="azure/gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello world"}
    ],
    api_key="your-azure-api-key",
    api_base="your-azure-endpoint"
)
```

**Key Features**:
- Provider-agnostic interface
- Fallback mechanisms
- Cost tracking
- Streaming support
- Caching capabilities

**Resources**:
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [GitHub Repository](https://github.com/BerriAI/litellm)

## MCP Development Tools

### Debugging & Testing

#### LangSmith

LangSmith provides tracing, evaluation, and debugging for LLM applications:

```python
import os
from langchain.callbacks.tracers.langchain_callback import LangChainTracer
from langchain.callbacks.tracers.langsmith import LangSmithConsoleCallbackHandler

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "your-langchain-api-key"
os.environ["LANGCHAIN_PROJECT"] = "MCP Testing"

# Add callback handler for console output
tracer = LangChainTracer(project_name="MCP Testing")
console_callback = LangSmithConsoleCallbackHandler()

# Use in your LangChain code
chat = ChatOpenAI(
    model_name="gpt-4",
    callbacks=[tracer, console_callback]
)
```

**Key Features**:
- Trace visualization
- Performance metrics
- Regression testing
- Annotation and feedback
- Prompt experimentation

**Resources**:
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangChain Integration](https://python.langchain.com/docs/langsmith/)

#### GPTCache

GPTCache provides caching for LLM responses to improve performance and reduce costs:

```python
from gptcache import cache
from gptcache.adapter import openai
from gptcache.manager import CacheBase, Manager, get_data_manager
from gptcache.processor.text import get_prompt

# Initialize cache
cache.init(
    pre_embedding_func=get_prompt,
    data_manager=get_data_manager()
)

# Use with OpenAI API (automatically caches)
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)
```

**Key Features**:
- Multiple storage backends
- Similarity search for responses
- Cache evaluation tools
- Custom cache strategies
- Cost estimation

**Resources**:
- [GPTCache Documentation](https://gptcache.readthedocs.io/)
- [GitHub Repository](https://github.com/zilliztech/gptcache)

## Frameworks for Specific MCP Use Cases

### Agent Frameworks

#### LangChain Agents

```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.chat_models import ChatOpenAI

# Define tools
@tool
def search_tool(query: str) -> str:
    """Search for information about a topic"""
    return f"Search results for {query}..."

@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression"""
    try:
        return str(eval(expression))
    except:
        return "Could not calculate expression"

# Create a model with MCP support
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Initialize the agent
agent = initialize_agent(
    tools=[search_tool, calculator],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)

# Run the agent
result = agent.invoke({"input": "What is 25*445 and who was the 35th US president?"})
print(result["output"])
```

#### AutoGen

Microsoft's AutoGen enables multi-agent conversations using MCP:

```python
import autogen

# Define agent configs
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "model": "gpt-4",
        "temperature": 0.7,
    }
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10
)

# Start a multi-agent conversation
user_proxy.initiate_chat(
    assistant,
    message="Design a simple weather app using Python and Streamlit"
)
```

### RAG Frameworks

#### LlamaIndex

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import ReActAgent

# Load and index documents
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Create query engine
query_engine = index.as_query_engine()

# Wrap as a tool
tools = [
    QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="document_search",
        description="Search for information in the documents"
    )
]

# Create an agent with the tool
agent = ReActAgent.from_tools(
    tools,
    llm="gpt-4",
    verbose=True
)

# Query the agent
response = agent.query("What information can I find in these documents?")
print(response)
```

## Selecting the Right Tools

When choosing MCP tools and frameworks, consider:

1. **Development Maturity**: How established and stable is the tool?
2. **Provider Support**: Which LLM providers are supported?
3. **Feature Completeness**: Does it cover all MCP features you need?
4. **Performance Overhead**: What's the impact on latency and throughput?
5. **Community Support**: Is there an active community and documentation?
6. **Integration Ease**: How easily does it fit into your tech stack?
7. **Scalability**: Can it handle your expected load?

## Getting Started Recommendations

For different use cases:

- **Simple Applications**: Start with provider SDKs (OpenAI, Anthropic)
- **Production Applications**: Consider LangChain or Semantic Kernel
- **Enterprise Integration**: Look at LlamaIndex or Haystack
- **Multi-provider Support**: Try LiteLLM or LangChain

## Further Resources

- [OpenAI MCP Documentation](https://platform.openai.com/docs/guides/text-generation/chat-completions-api)
- [LangChain AI Documentation](https://python.langchain.com/)
- [Semantic Kernel Handbook](https://learn.microsoft.com/en-us/semantic-kernel/)
- [Function Calling Standards](https://openai.com/blog/function-calling-and-other-api-updates)
- [AutoGen GitHub Repository](https://github.com/microsoft/autogen)