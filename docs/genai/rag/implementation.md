---
title: Implementing RAG Systems
sidebar_position: 3
description: Practical implementation of Retrieval-Augmented Generation systems with code examples
---

# Implementing RAG Systems

Retrieval-Augmented Generation (RAG) combines the knowledge access capabilities of retrieval systems with the generative abilities of large language models. This guide provides practical approaches to implementing RAG systems, with working code examples and best practices.

## RAG Architecture Overview

A typical RAG implementation involves these core components:

1. **Document Processing**: Convert documents into a format suitable for embedding
2. **Chunking**: Split documents into manageable pieces
3. **Embedding Generation**: Create vector representations of document chunks
4. **Vector Storage**: Store embeddings in a vector database for efficient retrieval
5. **Query Processing**: Convert user queries into embeddings and retrieve relevant chunks
6. **Augmented Generation**: Feed retrieved context and the original query to an LLM

![RAG Architecture](https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/static/img/retrieval.png)

## Setting Up a Basic RAG Pipeline

### 1. Document Processing

The first step is to load and process your documents:

```python
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Function to load documents based on file type
def load_document(file_path):
    """Load a document based on its file extension"""
    _, extension = os.path.splitext(file_path)
    
    if extension.lower() == '.pdf':
        loader = PyPDFLoader(file_path)
    elif extension.lower() == '.csv':
        loader = CSVLoader(file_path)
    elif extension.lower() in ['.txt', '.md', '.html']:
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")
        
    return loader.load()

# Example usage
documents = []
data_dir = "./data/"

for filename in os.listdir(data_dir):
    file_path = os.path.join(data_dir, filename)
    if os.path.isfile(file_path):
        try:
            doc = load_document(file_path)
            documents.extend(doc)
            print(f"Loaded {file_path}, {len(doc)} documents")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

print(f"Loaded {len(documents)} documents total")
```

### 2. Chunking Strategies

Splitting documents into appropriate chunks is crucial for effective retrieval:

```python
# Basic chunking with RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# Advanced chunking with semantic boundaries
from langchain.text_splitter import MarkdownHeaderTextSplitter

# For markdown files with headers
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# For a specific markdown document
with open('./data/documentation.md', 'r') as f:
    markdown_content = f.read()

md_header_splits = markdown_splitter.split_text(markdown_content)

# Further split by character
md_chunks = text_splitter.split_documents(md_header_splits)
```

#### Chunking Best Practices

1. **Chunk Size**: 
   - For general text: 300-1000 tokens works well
   - For technical content: Smaller chunks (200-500 tokens)
   - For narrative content: Larger chunks (800-1500 tokens)

2. **Chunk Overlap**: 
   - 10-20% overlap prevents information loss at boundaries
   - Higher overlap (up to 50%) for technical or dense information

3. **Semantic Chunking**:
   - Split on natural boundaries like paragraphs, sections, or headers
   - Use specialized splitters for specific formats (code, markdown, HTML)

```python
# Example of a more advanced chunking strategy
def chunk_with_metadata(documents, chunk_size=1000, chunk_overlap=200):
    """Chunk documents while preserving metadata and adding chunk position information"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.create_documents(
            [doc.page_content], 
            metadatas=[doc.metadata]
        )
        
        # Add chunk position metadata
        for i, chunk in enumerate(doc_chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_count"] = len(doc_chunks)
            chunks.append(chunk)
            
    return chunks
```

### 3. Generating Embeddings

Creating vector embeddings for your document chunks:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Using OpenAI embeddings
openai_embeddings = OpenAIEmbeddings()

# Using open-source alternative (sentence-transformers)
hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cuda'}  # Use GPU if available
)

# Generate embeddings for a sample chunk
sample_embedding = hf_embeddings.embed_query(chunks[0].page_content)
print(f"Embedding dimension: {len(sample_embedding)}")
```

### 4. Vector Storage

Setting up a vector database to store and query your embeddings:

```python
from langchain_community.vectorstores import Chroma, FAISS

# Create a persistent Chroma vector store
db = Chroma.from_documents(
    documents=chunks,
    embedding=hf_embeddings,
    persist_directory="./chroma_db"
)
db.persist()  # Save to disk

# Create an in-memory FAISS vector store (faster for larger datasets)
faiss_db = FAISS.from_documents(
    documents=chunks,
    embedding=hf_embeddings
)

# Optionally save and load FAISS index
faiss_db.save_local("faiss_index")
loaded_faiss_db = FAISS.load_local("faiss_index", hf_embeddings)
```

#### Comparison of Vector Databases

| Database | Strengths | Weaknesses | Best For |
|----------|-----------|------------|----------|
| Chroma | Easy setup, Python native | Less scalable | Development, small datasets |
| FAISS | Fast searches, memory-efficient | Limited metadata filtering | Medium-sized datasets, speed-critical apps |
| Pinecone | Fully managed, highly scalable | Paid service | Production, large-scale applications |
| Weaviate | Rich schema, hybrid search | More complex setup | Complex data relationships |
| Qdrant | Strong filtering, self-hostable | Newer in market | Production with complex filters |

```python
# Example with Pinecone (cloud vector DB)
from langchain_pinecone import PineconeVectorStore
import pinecone

# Initialize Pinecone
pinecone.init(
    api_key="YOUR_API_KEY",
    environment="us-west1-gcp"  # or your environment
)

index_name = "knowledge-base"

# Create index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=len(sample_embedding),
        metric="cosine"
    )

# Connect to the index
index = pinecone.Index(index_name)

# Create vector store
pinecone_store = PineconeVectorStore(
    index=index,
    embedding=hf_embeddings,
    text_key="text"
)

# Add documents
pinecone_store.add_documents(chunks)
```

### 5. Query Processing and Retrieval

Converting user queries into embeddings and retrieving relevant chunks:

```python
def retrieve_context(query, vectorstore, k=4):
    """Retrieve relevant context for a query"""
    # Get relevant documents from the vectorstore
    docs = vectorstore.similarity_search(query, k=k)
    
    # Format documents as context string
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    return context, docs

# Example usage
query = "How does the transformer architecture work?"
context, retrieved_docs = retrieve_context(query, faiss_db, k=4)

print(f"Retrieved {len(retrieved_docs)} documents")
print(f"First document: {retrieved_docs[0].page_content[:200]}...")
```

#### Advanced Retrieval Strategies

**Hybrid Search (Combining Dense and Sparse Retrieval)**:

```python
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Create a BM25 (keyword-based) retriever
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5  # Return top 5 results

# Create a dense retriever from vector store
dense_retriever = faiss_db.as_retriever(search_kwargs={"k": 5})

# Create an ensemble retriever that combines both
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.5, 0.5]
)

# Use the ensemble retriever
ensemble_results = ensemble_retriever.get_relevant_documents(query)
```

**Re-ranking Retrieved Results**:

```python
from sentence_transformers import CrossEncoder

# Initialize a cross-encoder for re-ranking
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_results(query, initial_results, top_k=3):
    """Re-rank retrieval results using a cross-encoder"""
    # Pair the query with each document
    pairs = [(query, doc.page_content) for doc in initial_results]
    
    # Get relevance scores
    scores = cross_encoder.predict(pairs)
    
    # Sort documents by score
    scored_results = list(zip(initial_results, scores))
    scored_results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top_k documents
    return [doc for doc, score in scored_results[:top_k]]

# Get initial results
initial_results = faiss_db.similarity_search(query, k=10)

# Re-rank to get the most relevant ones
reranked_docs = rerank_results(query, initial_results, top_k=4)
```

**Self-querying Retrieval**:

```python
from langchain.chains import create_self_query_chain
from langchain_openai import OpenAI

# Create a self-query chain that can interpret natural language filters
self_query_chain = create_self_query_chain(
    llm=OpenAI(temperature=0),
    vectorstore=faiss_db
)

# Use natural language that includes filter criteria
complex_query = "Find information about transformer architectures published after 2020"
generated_query, documents = self_query_chain.invoke({"query": complex_query})

print(f"Generated structured query: {generated_query}")
print(f"Retrieved {len(documents)} matching documents")
```

### 6. Augmented Generation

Combining the retrieved context with the user's query to generate better responses:

```python
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# Create a RAG prompt template
template = """Answer the question based only on the following context:

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Create a simple RAG chain
rag_chain = (
    {"context": lambda x: retrieve_context(x["question"], faiss_db)[0], "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Test the RAG chain
response = rag_chain.invoke({"question": "How does attention mechanism work in transformers?"})
print(response)
```

## Advanced RAG Implementations

### 1. Query Transformations

Improving retrieval by transforming the user's query:

```python
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# Query expansion template
query_expansion_template = """Given the user question below, expand it to be more specific and include related terms that might help in retrieving relevant documents.

User question: {question}

Expanded query:"""

expansion_prompt = PromptTemplate(template=query_expansion_template, input_variables=["question"])
query_expansion_chain = LLMChain(llm=llm, prompt=expansion_prompt)

# Generate multiple queries through hypothetical document embeddings (HyDE)
hyde_template = """Please write a passage that would answer the following question.
Question: {question}

Passage:"""

hyde_prompt = PromptTemplate(template=hyde_template, input_variables=["question"])
hyde_chain = LLMChain(llm=llm, prompt=hyde_prompt)

def retrieve_with_hyde(question, vectorstore, k=3):
    """Use HyDE to generate a hypothetical answer, and then use that to retrieve"""
    # Generate a hypothetical answer
    hypothetical_answer = hyde_chain.run(question=question)
    
    # Use the hypothetical answer to retrieve documents
    docs = vectorstore.similarity_search(hypothetical_answer, k=k)
    return docs

# Example usage
expanded_query = query_expansion_chain.run(question="What is RAG in AI?")
print(f"Original query: What is RAG in AI?")
print(f"Expanded query: {expanded_query}")

hyde_results = retrieve_with_hyde("What are the limitations of transformers?", faiss_db)
print(f"Retrieved {len(hyde_results)} documents using HyDE")
```

### 2. Multi-step RAG

Breaking the RAG process into multiple steps for complex queries:

```python
from typing import List, Dict
from langchain_core.documents import Document

def multi_step_rag(query: str, vectorstore) -> str:
    """
    Perform multi-step RAG for complex queries
    1. Decompose the query into sub-questions
    2. Answer each sub-question with RAG
    3. Synthesize the final answer
    """
    # 1. Decompose the query
    decompose_template = """Break down this complex question into 2-3 simpler sub-questions that would help answer the main question when combined.
    
    Complex question: {question}
    
    Sub-questions (output as a numbered list):"""
    
    decompose_prompt = PromptTemplate(template=decompose_template, input_variables=["question"])
    decompose_chain = LLMChain(llm=llm, prompt=decompose_prompt)
    
    sub_questions_text = decompose_chain.run(question=query)
    
    # Extract sub-questions (simple parsing, could be more robust)
    sub_questions = []
    for line in sub_questions_text.strip().split('\n'):
        if line.strip() and any(line.strip().startswith(str(i)) for i in range(1, 10)):
            # Remove the number and any special characters
            question = line.strip()
            for i in range(1, 10):
                prefix = f"{i}."
                if question.startswith(prefix):
                    question = question[len(prefix):].strip()
                    break
            sub_questions.append(question)
    
    # 2. Answer each sub-question
    sub_answers = []
    for i, sub_q in enumerate(sub_questions):
        print(f"Answering sub-question {i+1}: {sub_q}")
        # Use the basic RAG chain for each sub-question
        context, _ = retrieve_context(sub_q, vectorstore, k=3)
        
        sub_q_template = """Answer the question based only on the following context:
        
        {context}
        
        Question: {question}
        
        Answer:"""
        sub_q_prompt = PromptTemplate(template=sub_q_template, 
                                    input_variables=["context", "question"])
        sub_q_chain = LLMChain(llm=llm, prompt=sub_q_prompt)
        
        answer = sub_q_chain.run(context=context, question=sub_q)
        sub_answers.append({"question": sub_q, "answer": answer})
    
    # 3. Synthesize the final answer
    synthesis_template = """Based on the answers to several sub-questions, provide a comprehensive answer to the main question.
    
    Main question: {main_question}
    
    Sub-questions and answers:
    {sub_answers}
    
    Final comprehensive answer to the main question:"""
    
    # Format the sub-answers
    sub_answers_text = ""
    for i, sa in enumerate(sub_answers):
        sub_answers_text += f"Sub-question {i+1}: {sa['question']}\n"
        sub_answers_text += f"Answer {i+1}: {sa['answer']}\n\n"
    
    synthesis_prompt = PromptTemplate(template=synthesis_template, 
                                     input_variables=["main_question", "sub_answers"])
    synthesis_chain = LLMChain(llm=llm, prompt=synthesis_prompt)
    
    final_answer = synthesis_chain.run(main_question=query, sub_answers=sub_answers_text)
    
    return final_answer

# Example usage
complex_query = "How do transformer architectures compare to RNNs for sequence modeling, and what are their practical limitations?"
answer = multi_step_rag(complex_query, faiss_db)
print(f"Final answer: {answer}")
```

### 3. Conversational RAG with Memory

Adding conversation history to RAG systems:

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Create a memory object to store chat history
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create a conversational RAG chain
conversational_rag = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=faiss_db.as_retriever(search_kwargs={"k": 4}),
    memory=memory,
    verbose=True
)

# Example conversation
questions = [
    "How do transformers handle sequential data?",
    "What are their advantages over RNNs?",
    "Do they have any limitations for very long sequences?"
]

for question in questions:
    response = conversational_rag.invoke({"question": question})
    print(f"Question: {question}")
    print(f"Answer: {response['answer']}\n")
```

### 4. Dealing with Hallucinations

Implementing techniques to reduce hallucinations in RAG systems:

```python
def generate_with_fact_checking(query, context, llm):
    """Generate a response with built-in fact checking"""
    
    fact_check_prompt = f"""Answer the following question based ONLY on the provided context. 
    If the context doesn't contain enough information to answer confidently, acknowledge the limitations.
    
    Context:
    {context}
    
    Question: {query}
    
    Before providing your final answer, critically evaluate whether your response is fully supported by the context.
    
    Step 1: Write your initial answer based on the context.
    Step 2: Review each claim in your answer and verify it against the context.
    Step 3: If any claim isn't supported, revise your answer to remove or qualify that claim.
    Step 4: Provide your final, fact-checked answer.
    """
    
    response = llm.invoke(fact_check_prompt)
    return response

# Add confidence estimation
def generate_with_confidence(query, retrieved_docs, llm):
    """Generate a response with confidence estimation"""
    
    # Calculate a retrieval confidence score based on similarity
    similarities = [doc.metadata.get("score", 0.8) for doc in retrieved_docs if "score" in doc.metadata]
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.7
    
    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    
    confidence_prompt = f"""Answer the question based on the context provided. After your answer, include a confidence score (0-100%) that reflects how certain you are that your answer is fully supported by the provided context.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer the question first, then on a new line write "Confidence: X%" where X is your estimated confidence percentage.
    """
    
    response = llm.invoke(confidence_prompt)
    return response
```

## Evaluation and Monitoring

Implementing evaluation metrics for your RAG system:

```python
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.metrics.critique import harmfulness

from datasets import Dataset

# Prepare evaluation data
eval_data = {
    "question": ["How does the transformer architecture work?", 
                "What are the applications of RAG systems?"],
    "answer": ["Transformer architecture works by using self-attention mechanisms...",
              "RAG systems are used for knowledge-intensive tasks..."],
    "contexts": [
        [doc.page_content for doc in faiss_db.similarity_search("How does transformer architecture work?", k=3)],
        [doc.page_content for doc in faiss_db.similarity_search("What are applications of RAG?", k=3)]
    ],
    "ground_truths": [
        ["Transformer architecture uses self-attention to process all tokens in parallel..."],
        ["RAG systems are applied in question answering, conversational AI..."]
    ]
}

# Create a dataset for evaluation
eval_dataset = Dataset.from_dict(eval_data)

# Calculate metrics
result = evaluate(
    eval_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        harmfulness
    ]
)

print("Evaluation Results:")
print(f"Faithfulness: {result['faithfulness']:.4f}")
print(f"Answer Relevancy: {result['answer_relevancy']:.4f}")
print(f"Context Precision: {result['context_precision']:.4f}")
print(f"Context Recall: {result['context_recall']:.4f}")
print(f"Harmfulness: {result['harmfulness']:.4f}")
```

## Deployment Considerations

### Architecture Patterns

For production RAG systems, consider these architecture patterns:

1. **Async Processing Pipeline**:
   - Process documents asynchronously
   - Use message queues for document ingestion
   - Separate embedding generation from retrieval services

2. **Scalable Vector Search**:
   - Implement vector database sharding for large collections
   - Use read replicas for high-throughput applications
   - Consider ANN (Approximate Nearest Neighbor) for very large datasets

3. **Caching**:
   - Cache common queries and their retrieved contexts
   - Cache generated embeddings for frequent queries
   - Implement result caching with time-based invalidation

### Example Deployment with FastAPI

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

# Load environment variables
import dotenv
dotenv.load_dotenv()

# Initialize components
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

# Initialize models and vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.load_local("./faiss_index", embeddings)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create FastAPI app
app = FastAPI(title="RAG API")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 4

class RAGResponse(BaseModel):
    answer: str
    sources: list

@app.post("/rag", response_model=RAGResponse)
async def rag_endpoint(request: QueryRequest):
    try:
        # Get relevant documents
        docs = vector_store.similarity_search(request.query, k=request.top_k)
        
        # Extract context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate response
        prompt = f"""Answer the question based only on the following context:
        
        {context}
        
        Question: {request.query}
        
        Answer:"""
        
        response = llm.invoke(prompt)
        
        # Prepare sources for citation
        sources = []
        for doc in docs:
            source = {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            }
            sources.append(source)
            
        return {"answer": response.content, "sources": sources}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
```

## Performance Optimization

### Optimizing Retrieval Speed

```python
# Using HNSW index with FAISS for faster retrieval
import faiss

def create_optimized_faiss_index(embeddings_list, dimension):
    """Create an optimized FAISS index using HNSW algorithm"""
    # Convert embeddings to numpy array
    import numpy as np
    embeddings_array = np.array(embeddings_list).astype('float32')
    
    # Create HNSW index (Hierarchical Navigable Small World graphs)
    # M: number of connections per node (higher = more accurate but more memory)
    # efConstruction: construction time/accuracy tradeoff (higher = more accurate but slower build)
    M = 16  
    efConstruction = 200
    index = faiss.IndexHNSWFlat(dimension, M)
    index.hnsw.efConstruction = efConstruction
    index.hnsw.efSearch = 128  # Search time/accuracy tradeoff
    
    # Add vectors to the index
    index.add(embeddings_array)
    
    return index
```

### Optimizing Document Processing

```python
import concurrent.futures
from typing import List

def process_documents_parallel(file_paths: List[str], max_workers: int = 4):
    """Process documents in parallel"""
    documents = []
    
    # Process files in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a map of futures to file paths
        future_to_file = {
            executor.submit(load_document, file_path): file_path
            for file_path in file_paths
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                docs = future.result()
                documents.extend(docs)
                print(f"Processed {file_path}, {len(docs)} documents")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
    return documents
```

## Best Practices for Production RAG

1. **Document Processing**:
   - Extract text with layout preservation (for PDFs and structured documents)
   - Clean and normalize text to remove noise
   - Preserve document structure through metadata

2. **Chunking Strategy**:
   - Use semantic chunking when possible
   - Adjust chunk size based on content type and density
   - Include overlaps to avoid losing context at boundaries

3. **Retrieval Quality**:
   - Implement hybrid search (dense + sparse retrieval)
   - Use re-ranking for more relevant results
   - Consider query expansion techniques for better recall

4. **Response Generation**:
   - Include source attribution in responses
   - Implement fact-checking and hallucination reduction
   - Use chain-of-thought prompting for complex reasoning

5. **System Architecture**:
   - Separate ingestion and query pipelines
   - Implement caching at multiple levels
   - Set up monitoring for retrieval quality and response accuracy

6. **User Experience**:
   - Provide confidence scores with answers
   - Include citations and references to sources
   - Allow users to provide feedback on answer quality

By following these implementation patterns and best practices, you can build effective RAG systems that leverage both retrieved knowledge and the generative capabilities of LLMs to provide accurate, contextual responses to user queries.