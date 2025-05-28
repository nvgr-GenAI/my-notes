---
title: Introduction to RAG
sidebar_position: 1
description: Understanding Retrieval-Augmented Generation (RAG) and its core concepts
---

# Introduction to Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is an approach that enhances Large Language Models (LLMs) by supplementing their parametric knowledge with non-parametric, external information retrieved from a knowledge base. This technique allows models to access more current, accurate, and specialized information than what was available during their training.

## Why RAG Matters

LLMs are trained on vast amounts of text data, but they face several limitations:

1. **Knowledge Cutoff**: Models have no access to information beyond their training cutoff date
2. **Hallucinations**: They can generate plausible-sounding but incorrect information
3. **Limited Domain Expertise**: They may lack depth in specialized domains
4. **Source Attribution Challenges**: They cannot reliably cite sources for their knowledge

RAG addresses these limitations by:
- Providing access to up-to-date information
- Grounding responses in verifiable sources
- Enabling domain-specific knowledge retrieval
- Supporting source attribution and citations

## Core Components of RAG

![RAG Architecture Diagram](https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/static/img/retrieval.jpg)

### 1. Document Processing

Before retrieval can happen, source documents must be processed into a suitable format:

```python
import os
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader

def load_and_process_documents(directory_path):
    """
    Load documents from a directory with support for multiple file formats
    
    Args:
        directory_path: Path to the directory containing documents
        
    Returns:
        List of processed documents
    """
    documents = []
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            
        elif filename.endswith('.txt'):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
            
        elif filename.endswith('.md'):
            loader = UnstructuredMarkdownLoader(file_path)
            documents.extend(loader.load())
    
    print(f"Loaded {len(documents)} document chunks from {directory_path}")
    return documents
```

### 2. Chunking

Documents are split into manageable chunks for more effective retrieval:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=1000, chunk_overlap=100):
    """
    Split documents into chunks suitable for embedding
    
    Args:
        documents: List of documents to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks
```

Chunking strategies can vary based on the content type:
- **Paragraph-based**: Natural text boundaries for articles and narrative content
- **Fixed-size**: Consistent chunk size for uniform processing
- **Semantic-based**: Preserving meaning and context in chunks
- **Hierarchical**: Multiple levels of chunking for different retrieval needs

### 3. Embedding Generation

Converting text chunks into vector representations (embeddings):

```python
from langchain.embeddings import OpenAIEmbeddings
import numpy as np

def create_embeddings(chunks, embedding_model="text-embedding-ada-002"):
    """
    Generate embeddings for text chunks
    
    Args:
        chunks: List of text chunks
        embedding_model: Name of the embedding model to use
        
    Returns:
        List of text chunks with their embeddings
    """
    embeddings_model = OpenAIEmbeddings(model=embedding_model)
    
    # Generate embeddings for each chunk
    embedded_chunks = []
    for i, chunk in enumerate(chunks):
        embedding = embeddings_model.embed_query(chunk.page_content)
        embedded_chunks.append({
            "content": chunk.page_content,
            "embedding": embedding,
            "metadata": chunk.metadata
        })
        
        if (i + 1) % 100 == 0:
            print(f"Embedded {i + 1} chunks")
    
    return embedded_chunks
```

### 4. Vector Database Storage

Storing embeddings efficiently for similarity search:

```python
import chromadb
from langchain.vectorstores import Chroma

def create_vector_store(chunks, embeddings, persist_directory=None):
    """
    Create and populate a vector store with document chunks
    
    Args:
        chunks: List of text chunks to store
        embeddings: Embedding function to use
        persist_directory: Optional directory to persist the database
        
    Returns:
        Vector store object
    """
    # Create and populate the vector store
    if persist_directory:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        # Persist to disk
        vector_store.persist()
    else:
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings
        )
    
    return vector_store
```

Popular vector database options include:
- **Chroma**: Open-source, easy to use for small to medium projects
- **Pinecone**: Cloud-based, scalable service with strong performance 
- **Weaviate**: Open-source vector search with schema capabilities
- **Milvus**: Distributed database for very large-scale applications
- **Qdrant**: Self-hostable with rich filtering capabilities

### 5. Query Processing

Transforming user queries for effective retrieval:

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI

def process_and_expand_query(query, llm_model="gpt-3.5-turbo"):
    """
    Process a user query and generate alternative formulations
    
    Args:
        query: Original user query
        llm_model: LLM to use for query expansion
        
    Returns:
        List of query variations
    """
    llm = ChatOpenAI(model_name=llm_model, temperature=0)
    
    # Generate alternative query formulations
    prompt = f"""
    Generate three different versions of the following query. Each version should capture the same information need but use different wording, perspective, or level of specificity.
    
    Original query: "{query}"
    
    Provide exactly three alternatives, each on a separate line, numbered 1-3.
    """
    
    response = llm.predict(prompt)
    
    # Extract the alternative queries
    alternatives = []
    for line in response.strip().split('\n'):
        if line and any(line.startswith(f"{i}") for i in range(1, 4)):
            # Remove the leading number and any special characters
            clean_line = line[line.find(" ")+1:].strip()
            # Remove quotes if present
            clean_line = clean_line.strip('"').strip("'")
            alternatives.append(clean_line)
    
    # Always include the original query
    all_queries = [query] + alternatives
    print(f"Expanded query into {len(all_queries)} variations")
    return all_queries
```

### 6. Retrieval

Finding the most relevant documents for a user query:

```python
def retrieve_relevant_chunks(vector_store, query, k=5):
    """
    Retrieve the most relevant document chunks for a query
    
    Args:
        vector_store: Vector store containing document chunks
        query: User query
        k: Number of chunks to retrieve
        
    Returns:
        List of relevant document chunks
    """
    # Perform similarity search
    results = vector_store.similarity_search_with_score(query, k=k)
    
    # Format the results
    retrieved_chunks = []
    for doc, score in results:
        retrieved_chunks.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "relevance_score": score
        })
    
    return retrieved_chunks
```

Advanced retrieval techniques:
- **Hybrid search**: Combining semantic and keyword search
- **Re-ranking**: Using a secondary model to refine search results
- **Ensemble methods**: Combining results from multiple retrieval strategies
- **Contextual compression**: Extracting only the relevant parts of documents

### 7. Response Generation

Combining retrieved information with the LLM to generate a response:

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

def generate_rag_response(query, retrieved_chunks, model_name="gpt-4"):
    """
    Generate a response based on the retrieved information
    
    Args:
        query: User query
        retrieved_chunks: List of relevant document chunks
        model_name: Name of the LLM to use
        
    Returns:
        Generated response
    """
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    
    # Prepare the context from retrieved chunks
    context = "\n\n".join([f"Document {i+1}:\n{chunk['content']}" for i, chunk in enumerate(retrieved_chunks)])
    
    # Construct the prompt
    system_message = """
    You are a helpful assistant that answers questions based on the provided information.
    When answering, follow these rules:
    1. Only use information from the provided documents
    2. If the documents don't contain relevant information, say "I don't have enough information to answer this question"
    3. Always cite your sources by referring to the document number (e.g., "According to Document 1...")
    4. Do not make up information or use knowledge beyond what's in the provided documents
    """
    
    human_message = f"""
    Question: {query}
    
    Relevant Information:
    {context}
    
    Please answer the question based solely on the provided information.
    """
    
    # Generate the response
    messages = [SystemMessage(content=system_message), HumanMessage(content=human_message)]
    response = llm.predict_messages(messages)
    
    return response.content
```

## Putting It All Together: Basic RAG Pipeline

Here's how to implement a complete RAG pipeline:

```python
class RAGPipeline:
    def __init__(self, documents_path, persist_directory=None):
        """
        Initialize the RAG pipeline
        
        Args:
            documents_path: Path to documents directory
            persist_directory: Optional directory to persist vector store
        """
        # Initialize embedding model
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        # Set up document processing pipeline
        documents = load_and_process_documents(documents_path)
        chunks = chunk_documents(documents)
        
        # Create and store the vector database
        self.vector_store = create_vector_store(chunks, self.embeddings, persist_directory)
        
        # Initialize LLM
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    
    def query(self, user_query, k=5):
        """
        Process a query through the RAG pipeline
        
        Args:
            user_query: Question or query from the user
            k: Number of relevant chunks to retrieve
            
        Returns:
            Generated answer based on retrieved information
        """
        # Process and expand the query
        expanded_queries = process_and_expand_query(user_query)
        
        # Retrieve relevant chunks for each query variation
        all_results = []
        for query in expanded_queries:
            results = retrieve_relevant_chunks(self.vector_store, query, k)
            all_results.extend(results)
        
        # Remove duplicates and select top k most relevant chunks
        unique_results = {result["content"]: result for result in all_results}
        top_results = sorted(unique_results.values(), key=lambda x: x["relevance_score"])[:k]
        
        # Generate response
        response = generate_rag_response(user_query, top_results)
        
        return {
            "query": user_query,
            "retrieved_chunks": top_results,
            "response": response
        }
```

## RAG vs. Fine-Tuning

When deciding between RAG and fine-tuning for your application, consider:

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Up-to-date information** | ✅ Can use real-time information | ❌ Limited to training data |
| **Development time** | ✅ Faster to implement | ❌ Requires training time |
| **Cost** | ✅ Lower compute cost | ❌ Higher compute cost |
| **Adaptability** | ✅ Easy to update knowledge | ❌ Requires retraining |
| **Transparency** | ✅ Sources can be cited | ❌ "Black box" knowledge |
| **Performance** | ❌ Higher latency | ✅ Lower latency |
| **Domain expertise** | ✅ Can use domain-specific docs | ✅ Can learn domain patterns |
| **Consistency** | ❌ May be inconsistent | ✅ More consistent outputs |

The best approach often combines both: fine-tune for domain-specific language patterns and use RAG for up-to-date information.

## Challenges and Limitations

While powerful, RAG systems face several challenges:

1. **Retrieval Quality**: Results are only as good as the retrieval mechanism
2. **Context Window Limitations**: LLMs can only process a finite amount of retrieved context
3. **Source Document Quality**: Poor documents lead to poor responses
4. **Computational Overhead**: Additional processing time for retrieval steps
5. **Integration Complexity**: More complex than simple LLM applications

## Advanced RAG Techniques

As RAG continues to evolve, several advanced techniques are emerging:

### Hypothetical Document Embeddings (HyDE)

Generate a hypothetical answer first, then use it for retrieval:

```python
def hyde_retrieval(query, vector_store, llm):
    """
    Hypothetical Document Embedding retrieval method
    
    Args:
        query: User query
        vector_store: Vector store for retrieval
        llm: Language model for generating hypothetical answer
        
    Returns:
        Retrieved documents
    """
    # Step 1: Generate a hypothetical document/answer
    prompt = f"""
    Given the question: "{query}"
    
    Write a detailed paragraph that could serve as the perfect answer to this question.
    This is a hypothetical answer and will be used to help retrieve relevant information.
    """
    
    hypothetical_doc = llm.predict(prompt)
    
    # Step 2: Use the hypothetical document for retrieval instead of the query
    results = vector_store.similarity_search(hypothetical_doc, k=5)
    
    return results
```

### Self-RAG

The model learns to decide when to retrieve and when to use its parametric knowledge:

```python
def self_rag_query(query, vector_store, llm):
    """
    Self-RAG approach where the model decides whether to retrieve
    
    Args:
        query: User query
        vector_store: Vector store for retrieval
        llm: Language model
        
    Returns:
        Generated response
    """
    # Step 1: Ask the model if retrieval is needed
    retrieval_decision_prompt = f"""
    Question: {query}
    
    Do I need to retrieve external information to answer this question accurately, 
    or can I answer it based on general knowledge? Respond with just "Retrieve" 
    or "No retrieval needed" and a very brief explanation.
    """
    
    retrieval_decision = llm.predict(retrieval_decision_prompt)
    
    # Step 2: Retrieve if needed
    if "Retrieve" in retrieval_decision:
        retrieved_docs = vector_store.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        
        answer_prompt = f"""
        Question: {query}
        
        Here is some relevant information:
        {context}
        
        Please answer the question based on this information.
        """
    else:
        answer_prompt = f"""
        Question: {query}
        
        Please answer this question based on your knowledge.
        """
    
    # Step 3: Generate the final answer
    answer = llm.predict(answer_prompt)
    
    return {
        "query": query,
        "retrieval_decision": retrieval_decision,
        "answer": answer
    }
```

As RAG systems continue to evolve, they increasingly represent the best of both worlds: combining the broad knowledge and capabilities of LLMs with the accuracy, recency, and specificity of external knowledge sources. This makes them one of the most promising approaches for building reliable AI applications across domains.