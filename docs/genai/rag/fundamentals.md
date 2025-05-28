---
title: RAG Fundamentals
sidebar_position: 1
description: Introduction to Retrieval-Augmented Generation (RAG) architecture and components
---

# Retrieval-Augmented Generation (RAG) Fundamentals

Retrieval-Augmented Generation (RAG) is an architecture that combines the strengths of retrieval-based systems and generative models to produce more accurate, factual, and contextually relevant outputs. RAG models first retrieve relevant documents or pieces of information from a knowledge base, then use this information to condition a language model to generate a response.

## Why RAG Matters

Traditional Large Language Models (LLMs) face several challenges:

1. **Knowledge cutoff**: They only know information up to their training cutoff date
2. **Hallucinations**: They can generate plausible-sounding but incorrect information
3. **Verifiability**: Their responses lack clear sources or citations
4. **Cost**: Continually retraining models with new knowledge is expensive

RAG addresses these challenges by:

1. **Accessing up-to-date information**: By retrieving from regularly updated knowledge sources
2. **Grounding responses in evidence**: Using retrieved information to reduce hallucinations
3. **Providing attribution**: Enabling traceability to original sources
4. **Efficient knowledge updating**: Only updating the retrieval corpus, not the model weights

## Core Components of a RAG System

![RAG Architecture Overview](https://miro.medium.com/v2/resize:fit:1400/1*TG-1mQGN96tKGFtOTTE2kA.jpeg)

### 1. Knowledge Base and Indexing

The knowledge base stores the information that the RAG system can retrieve from:

```python
import os
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Any

class KnowledgeBase:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.documents = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def add_documents_from_directory(self, directory_path: str, glob_pattern: str = "**/*.txt"):
        """Load documents from a directory matching the glob pattern"""
        loader = DirectoryLoader(
            directory_path, 
            glob=glob_pattern,
            loader_cls=TextLoader
        )
        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)
        self.documents.extend(chunks)
        return len(chunks)
    
    def add_pdf_document(self, file_path: str):
        """Load and process a PDF document"""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)
        self.documents.extend(chunks)
        return len(chunks)
    
    def add_text(self, text: str, metadata: Dict[str, Any] = None):
        """Add raw text with optional metadata"""
        metadata = metadata or {}
        doc = Document(page_content=text, metadata=metadata)
        chunks = self.text_splitter.split_documents([doc])
        self.documents.extend(chunks)
        return len(chunks)
    
    def get_document_count(self) -> int:
        """Get the total number of document chunks in the knowledge base"""
        return len(self.documents)
```

### 2. Vector Store for Embeddings

To enable efficient retrieval, documents are converted to vector embeddings:

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

class VectorStore:
    def __init__(self, embedding_model_name="text-embedding-3-small"):
        """Initialize vector store with specified embedding model"""
        self.embeddings = OpenAIEmbeddings(model=embedding_model_name)
        self.vector_store = None
    
    def create_from_documents(self, documents: List[Document], persist_directory: str = None):
        """Create a vector store from documents"""
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        if persist_directory:
            self.vector_store.persist()
        return self.vector_store
    
    def load_existing(self, persist_directory: str):
        """Load an existing vector store from disk"""
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        return self.vector_store
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve the k most similar documents to the query"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Create or load one first.")
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """Retrieve the k most similar documents with relevance scores"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Create or load one first.")
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to an existing vector store"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Create or load one first.")
        self.vector_store.add_documents(documents)
```

### 3. Retriever

The retriever is responsible for finding relevant documents from the vector store:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chat_models import ChatOpenAI

class EnhancedRetriever:
    def __init__(self, vector_store, use_compression=False, llm_model_name="gpt-3.5-turbo"):
        """
        Initialize retriever with options for enhanced retrieval
        
        Args:
            vector_store: Initialized vector store
            use_compression: Whether to use contextual compression
            llm_model_name: Model to use for compression if enabled
        """
        self.vector_store = vector_store
        self.base_retriever = vector_store.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        if use_compression:
            # Initialize LLM for compression
            llm = ChatOpenAI(model=llm_model_name)
            # Create compressor that extracts most relevant information
            compressor = LLMChainExtractor.from_llm(llm)
            # Create compression retriever
            self.retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.base_retriever
            )
        else:
            self.retriever = self.base_retriever
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query to retrieve documents for
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        # Update search parameters
        if hasattr(self.retriever, "search_kwargs"):
            self.retriever.search_kwargs["k"] = top_k
        elif hasattr(self.retriever, "base_retriever") and hasattr(self.retriever.base_retriever, "search_kwargs"):
            self.retriever.base_retriever.search_kwargs["k"] = top_k
            
        return self.retriever.get_relevant_documents(query)
    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Document]:
        """
        Perform hybrid search (combining keyword and semantic search)
        This is a simplified implementation - actual hybrid search would use BM25 or similar
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            alpha: Weight between semantic (0) and keyword matching (1)
            
        Returns:
            Combined list of retrieved documents
        """
        # Simple implementation - in production, use a proper hybrid search
        # This would typically combine BM25 with vector search
        semantic_results = self.retrieve(query, top_k=top_k)
        
        # In a real implementation, you would:
        # 1. Perform keyword search (e.g., with ElasticSearch/BM25)
        # 2. Perform vector search
        # 3. Combine results with a ranking formula
        
        return semantic_results
```

### 4. Prompt Construction

Creating effective prompts that incorporate retrieved information:

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class RAGPromptManager:
    def __init__(self):
        """Initialize with default prompt templates"""
        # Basic RAG prompt
        self.basic_rag_template = """Answer the question based on the context below. If the question cannot be answered using the information provided, say "I don't have enough information to answer the question."
        
Context:
{context}

Question: {question}

Answer:"""
        
        # Template that encourages citing sources
        self.source_attribution_template = """Answer the question based on the retrieved context below. For any information used in your answer, indicate which source(s) it came from using [1], [2], etc. If the question cannot be answered using the information provided, say "I don't have enough information to answer the question."

Context:
{context}

Question: {question}

Answer (with source attributions):"""
        
        # Template for more analytical responses
        self.analytical_template = """Based on the context below, provide a thorough analysis in response to the question. Consider multiple perspectives, identify key factors, and highlight any limitations in the available information.

Context:
{context}

Question: {question}

Analytical Response:"""
        
        # Default template to use
        self.current_template = self.basic_rag_template
    
    def get_prompt(self, template_style="basic"):
        """Get a prompt template by style"""
        if template_style == "basic":
            self.current_template = self.basic_rag_template
        elif template_style == "attribution":
            self.current_template = self.source_attribution_template
        elif template_style == "analytical":
            self.current_template = self.analytical_template
        else:
            raise ValueError(f"Unknown template style: {template_style}")
            
        return PromptTemplate(
            input_variables=["context", "question"],
            template=self.current_template
        )
    
    def set_custom_template(self, template: str):
        """Set a custom prompt template"""
        self.current_template = template
        return PromptTemplate(
            input_variables=["context", "question"],
            template=self.current_template
        )
    
    def format_retrieved_context(self, docs: List[Document]) -> str:
        """Format retrieved documents into a context string with source attributions"""
        formatted_context = ""
        
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            source = doc.metadata.get("source", f"Document {i}")
            formatted_context += f"[{i}] Source: {source}\n{content}\n\n"
            
        return formatted_context
```

### 5. Generator (LLM)

The final component that produces the response based on the query and retrieved context:

```python
from langchain.schema import BaseRetriever
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

class RAGGenerator:
    def __init__(self, retriever: BaseRetriever, model_name="gpt-3.5-turbo", temperature=0.2):
        """
        Initialize RAG generator
        
        Args:
            retriever: Document retriever component
            model_name: LLM model to use for generation
            temperature: Sampling temperature for generation
        """
        self.retriever = retriever
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.prompt_manager = RAGPromptManager()
    
    def setup_rag_chain(self, prompt_style="basic"):
        """
        Create a RAG chain using LangChain's LCEL (LangChain Expression Language)
        
        Args:
            prompt_style: Style of prompt to use
        """
        # Get appropriate prompt template
        prompt = self.prompt_manager.get_prompt(prompt_style)
        
        # Define context retrieval and preparation function
        def retrieve_and_format_context(query):
            docs = self.retriever.retrieve(query)
            return self.prompt_manager.format_retrieved_context(docs)
        
        # Create the RAG chain
        self.chain = (
            {"context": retrieve_and_format_context, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return self.chain
    
    def answer_query(self, query: str, prompt_style="basic") -> str:
        """
        Process a query through the RAG pipeline
        
        Args:
            query: User query
            prompt_style: Style of prompt to use
            
        Returns:
            Generated answer
        """
        # Setup the chain
        chain = self.setup_rag_chain(prompt_style)
        
        # Run the chain
        response = chain.invoke(query)
        
        return response
```

## Putting It All Together: Basic RAG Pipeline

Here's how to integrate all the components into a complete RAG system:

```python
class RAGSystem:
    def __init__(
        self,
        embedding_model="text-embedding-3-small",
        llm_model="gpt-3.5-turbo",
        chunk_size=1000,
        chunk_overlap=200,
        use_compression=False
    ):
        """
        Initialize a complete RAG system
        
        Args:
            embedding_model: Model for creating embeddings
            llm_model: Language model for generation
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            use_compression: Whether to use retrieval compression
        """
        # Initialize knowledge base
        self.kb = KnowledgeBase(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Initialize vector store
        self.vector_store = VectorStore(embedding_model_name=embedding_model)
        
        # Retriever will be initialized after adding documents
        self.retriever = None
        
        # Configuration for generator
        self.llm_model = llm_model
        self.use_compression = use_compression
        
        # The generator will be initialized after setting up the retriever
        self.generator = None
    
    def add_documents(self, directory_path=None, pdf_files=None, texts=None, persist_dir=None):
        """
        Add documents to the knowledge base
        
        Args:
            directory_path: Path to directory of text files
            pdf_files: List of PDF file paths
            texts: List of (text, metadata) tuples
            persist_dir: Directory to persist vector store
        """
        doc_count = 0
        
        # Add documents from directory
        if directory_path:
            doc_count += self.kb.add_documents_from_directory(directory_path)
        
        # Add PDF documents
        if pdf_files:
            for pdf_file in pdf_files:
                doc_count += self.kb.add_pdf_document(pdf_file)
        
        # Add raw texts with metadata
        if texts:
            for text, metadata in texts:
                doc_count += self.kb.add_text(text, metadata)
        
        # Create vector store from documents
        self.vector_store.create_from_documents(
            self.kb.documents,
            persist_directory=persist_dir
        )
        
        # Initialize retriever
        self.retriever = EnhancedRetriever(
            self.vector_store,
            use_compression=self.use_compression,
            llm_model_name=self.llm_model
        )
        
        # Initialize generator with retriever
        self.generator = RAGGenerator(
            self.retriever,
            model_name=self.llm_model
        )
        
        return doc_count
    
    def load_existing(self, persist_dir):
        """Load an existing vector store from disk"""
        self.vector_store.load_existing(persist_dir)
        
        # Initialize retriever
        self.retriever = EnhancedRetriever(
            self.vector_store,
            use_compression=self.use_compression,
            llm_model_name=self.llm_model
        )
        
        # Initialize generator with retriever
        self.generator = RAGGenerator(
            self.retriever,
            model_name=self.llm_model
        )
    
    def answer_question(self, question, prompt_style="basic"):
        """
        Generate an answer to a question using RAG
        
        Args:
            question: User query
            prompt_style: Style of prompt to use
            
        Returns:
            Generated answer
        """
        if not self.generator:
            raise ValueError("System not initialized. Add documents or load existing vector store first.")
            
        return self.generator.answer_query(question, prompt_style=prompt_style)
```

## Common RAG Architectures and Variations

### 1. Basic RAG

The simplest form, retrieving documents and directly using them as context:

```python
# Example usage of basic RAG
rag_system = RAGSystem(
    embedding_model="text-embedding-3-small",
    llm_model="gpt-3.5-turbo"
)

# Add documents
rag_system.add_documents(
    directory_path="./data/documents/",
    pdf_files=["./data/reports/annual_report_2023.pdf"]
)

# Answer a question
answer = rag_system.answer_question(
    "What were the key financial metrics for Q3 2023?"
)
print(answer)
```

### 2. Hypothetical Document Embeddings (HyDE)

HyDE first generates a hypothetical answer, then uses it for retrieval:

```python
from langchain.chains import HypotheticalDocumentEmbedder

class HyDERAGSystem(RAGSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_hyde = True
        
    def setup_hyde(self):
        """Set up Hypothetical Document Embedder"""
        # Initialize LLM for generating hypothetical documents
        llm = ChatOpenAI(model=self.llm_model)
        
        # Template for generating hypothetical documents
        hyde_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Given the following question, generate a passage that could contain the answer.
            Question: {question}
            Passage:"""
        )
        
        # Create the HyDE retriever
        self.hyde_embedder = HypotheticalDocumentEmbedder(
            llm_chain=LLMChain(llm=llm, prompt=hyde_prompt),
            base_embeddings=self.vector_store.embeddings
        )
        
        return self.hyde_embedder
    
    def answer_with_hyde(self, question, prompt_style="basic"):
        """Answer using the HyDE approach"""
        # First, generate a hypothetical answer
        hyde_embedder = self.setup_hyde()
        
        # Get document embeddings using the hypothetical document
        query_vector = hyde_embedder.embed_query(question)
        
        # Use the vector store to find similar documents using the generated embedding
        docs = self.vector_store.vector_store.similarity_search_by_vector(
            query_vector, k=5
        )
        
        # Format retrieved context
        context = self.generator.prompt_manager.format_retrieved_context(docs)
        
        # Get appropriate prompt template
        prompt = self.generator.prompt_manager.get_prompt(prompt_style)
        
        # Format the prompt with context and question
        formatted_prompt = prompt.format(context=context, question=question)
        
        # Generate answer with the LLM
        response = self.generator.llm.predict(formatted_prompt)
        
        return response
```

### 3. Multi-step Retrieval and Reasoning

More complex architecture that breaks down queries and performs iterative retrieval:

```python
class MultiStepRAG(RAGSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def decompose_question(self, question):
        """Break down a complex question into sub-questions"""
        decomposition_prompt = f"""
        Break down the following complex question into 2-3 simpler sub-questions that would help answer the overall question when combined.
        
        Complex question: {question}
        
        Format your response as a JSON array of strings, each containing one sub-question.
        """
        
        llm = ChatOpenAI(model=self.llm_model, temperature=0.2)
        response = llm.predict(decomposition_prompt)
        
        try:
            # Parse the response as JSON
            import json
            sub_questions = json.loads(response)
            return sub_questions
        except:
            # Fallback if parsing fails
            return [question]
    
    def answer_multi_step(self, question):
        """Answer a question using multi-step retrieval and reasoning"""
        # Step 1: Decompose the question
        sub_questions = self.decompose_question(question)
        
        # Step 2: Answer each sub-question
        sub_answers = []
        for sub_q in sub_questions:
            answer = self.answer_question(sub_q)
            sub_answers.append({"question": sub_q, "answer": answer})
        
        # Step 3: Synthesize a final answer
        synthesis_prompt = f"""
        Based on the following sub-questions and their answers, provide a comprehensive answer to the main question.
        
        Main question: {question}
        
        Sub-questions and answers:
        {json.dumps(sub_answers, indent=2)}
        
        Synthesized answer:
        """
        
        llm = ChatOpenAI(model=self.llm_model, temperature=0.2)
        final_answer = llm.predict(synthesis_prompt)
        
        return {
            "final_answer": final_answer,
            "sub_questions": sub_answers
        }
```

## Evaluating RAG Systems

Proper evaluation is crucial for developing effective RAG systems:

```python
from datasets import load_dataset
import random
import numpy as np
from rouge_score import rouge_scorer
from typing import List, Dict, Any

class RAGEvaluator:
    def __init__(self):
        """Initialize RAG evaluator with evaluation metrics"""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def create_test_set(self, dataset_name=None, num_samples=100):
        """Create or load a test set for evaluation"""
        if dataset_name:
            # Load pre-made dataset (e.g., from Hugging Face)
            dataset = load_dataset(dataset_name, split="test")
            # Take a sample
            test_samples = dataset.select(range(min(num_samples, len(dataset))))
        else:
            # This would be replaced with your actual test set creation logic
            test_samples = [
                {"question": "What is RAG in AI?", "answer": "RAG (Retrieval-Augmented Generation) combines retrieval and generation for more accurate AI responses."}
                # Add more samples
            ]
        
        return test_samples
    
    def evaluate_relevance(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate relevance of retrieved documents
        
        Args:
            query: User query
            retrieved_docs: List of retrieved documents
            
        Returns:
            Dictionary of relevance metrics
        """
        # This is a simplified implementation
        # In a real system, you would use human judgments or more sophisticated metrics
        
        # Initialize LLM for evaluation
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        relevance_scores = []
        
        for i, doc in enumerate(retrieved_docs):
            content = doc.page_content
            
            # Create evaluation prompt
            eval_prompt = f"""
            On a scale of 1-10, rate how relevant the following document is to the query.
            
            Query: {query}
            
            Document: {content}
            
            Provide only a numeric rating from 1-10, where:
            1: Completely irrelevant
            10: Perfectly relevant and directly answers the query
            
            Rating:
            """
            
            # Get rating from LLM
            response = llm.predict(eval_prompt).strip()
            
            try:
                # Extract numeric rating
                rating = float(response.split()[0])
                relevance_scores.append(rating)
            except:
                # Fallback if parsing fails
                relevance_scores.append(5.0)  # Neutral score
        
        # Calculate metrics
        results = {
            "avg_relevance": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
            "max_relevance": max(relevance_scores) if relevance_scores else 0,
            "min_relevance": min(relevance_scores) if relevance_scores else 0
        }
        
        return results
    
    def evaluate_accuracy(self, generated_answer: str, reference_answer: str) -> Dict[str, float]:
        """
        Evaluate accuracy of the generated answer against reference
        
        Args:
            generated_answer: Answer generated by the RAG system
            reference_answer: Reference/ground truth answer
            
        Returns:
            Dictionary of accuracy metrics
        """
        # Calculate ROUGE scores
        rouge_scores = self.rouge_scorer.score(reference_answer, generated_answer)
        
        # Convert to dictionary
        results = {
            "rouge1": rouge_scores['rouge1'].fmeasure,
            "rouge2": rouge_scores['rouge2'].fmeasure,
            "rougeL": rouge_scores['rougeL'].fmeasure
        }
        
        return results
    
    def evaluate_hallucination(self, generated_answer: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the generated answer for hallucinations (content not in retrieved docs)
        
        Args:
            generated_answer: Answer generated by the RAG system
            retrieved_docs: Documents retrieved and used for generation
            
        Returns:
            Hallucination assessment
        """
        # Concatenate all retrieved document content
        all_content = " ".join([doc.page_content for doc in retrieved_docs])
        
        # Initialize LLM for evaluation
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        # Create evaluation prompt
        eval_prompt = f"""
        Determine if the generated answer contains information not present in the retrieved documents.
        
        Generated answer: {generated_answer}
        
        Retrieved documents content: {all_content}
        
        Provide:
        1. A hallucination score from 1-10 where:
           1: No hallucination, answer is completely supported by documents
           10: Severe hallucination, major claims have no support in documents
           
        2. Specific examples of any hallucinated information
        
        Format your response as a JSON object with fields "score" and "examples".
        """
        
        response = llm.predict(eval_prompt)
        
        try:
            # Parse the response as JSON
            import json
            hallucination_assessment = json.loads(response)
            return hallucination_assessment
        except:
            # Fallback if parsing fails
            return {
                "score": 5.0,  # Neutral score
                "examples": ["Failed to parse hallucination assessment"]
            }
    
    def run_evaluation(self, rag_system, test_samples):
        """
        Run comprehensive evaluation on a RAG system
        
        Args:
            rag_system: The RAG system to evaluate
            test_samples: List of test cases with questions and reference answers
            
        Returns:
            Evaluation results
        """
        results = []
        
        for sample in test_samples:
            question = sample["question"]
            reference_answer = sample["answer"]
            
            # Process with RAG system
            generated_answer = rag_system.answer_question(question)
            
            # Get retrieved documents
            retrieved_docs = rag_system.retriever.retrieve(question)
            
            # Evaluate
            relevance_metrics = self.evaluate_relevance(question, retrieved_docs)
            accuracy_metrics = self.evaluate_accuracy(generated_answer, reference_answer)
            hallucination_assessment = self.evaluate_hallucination(generated_answer, retrieved_docs)
            
            # Compile results for this sample
            sample_result = {
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "relevance_metrics": relevance_metrics,
                "accuracy_metrics": accuracy_metrics,
                "hallucination_assessment": hallucination_assessment
            }
            
            results.append(sample_result)
        
        # Calculate aggregate metrics
        aggregates = {
            "avg_relevance": sum(r["relevance_metrics"]["avg_relevance"] for r in results) / len(results) if results else 0,
            "avg_rouge1": sum(r["accuracy_metrics"]["rouge1"] for r in results) / len(results) if results else 0,
            "avg_rouge2": sum(r["accuracy_metrics"]["rouge2"] for r in results) / len(results) if results else 0,
            "avg_rougeL": sum(r["accuracy_metrics"]["rougeL"] for r in results) / len(results) if results else 0,
            "avg_hallucination_score": sum(r["hallucination_assessment"]["score"] for r in results) / len(results) if results else 0
        }
        
        return {
            "sample_results": results,
            "aggregate_metrics": aggregates
        }
```

## Optimizing RAG Systems

Here are some techniques to improve RAG performance:

### 1. Query Transformations

```python
class QueryTransformer:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.2)
    
    def expand_query(self, query: str) -> str:
        """
        Expand a query with additional related terms
        
        Args:
            query: Original user query
            
        Returns:
            Expanded query
        """
        prompt = f"""
        Expand the following search query by adding related terms that could help retrieve more relevant information. 
        Keep the original query intact and add 3-5 related terms or synonyms.
        
        Original query: {query}
        
        Expanded query:
        """
        
        expanded = self.llm.predict(prompt)
        return expanded
    
    def rewrite_for_retrieval(self, query: str) -> str:
        """
        Rewrite a query specifically for retrieval effectiveness
        
        Args:
            query: Original user query
            
        Returns:
            Rewritten query optimized for retrieval
        """
        prompt = f"""
        Rewrite the following query to make it more effective for retrieval from a document database.
        Focus on key entities, concepts, and specific terminology. Remove unnecessary words and conversational elements.
        
        Original query: {query}
        
        Rewritten query:
        """
        
        rewritten = self.llm.predict(prompt)
        return rewritten
    
    def generate_query_variations(self, query: str, num_variations: int = 3) -> List[str]:
        """
        Generate multiple variations of a query for ensemble retrieval
        
        Args:
            query: Original user query
            num_variations: Number of variations to generate
            
        Returns:
            List of query variations
        """
        prompt = f"""
        Generate {num_variations} different variations of the following query. Each variation should
        capture the same information need but be phrased differently.
        
        Original query: {query}
        
        Provide the variations as a JSON array of strings.
        """
        
        response = self.llm.predict(prompt)
        
        try:
            # Parse the response as JSON
            import json
            variations = json.loads(response)
            return variations
        except:
            # Fallback if parsing fails
            return [query]
```

### 2. Dynamic Retrieval

```python
class DynamicRetriever:
    def __init__(self, vector_store, model_name="gpt-3.5-turbo"):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.2)
        self.query_transformer = QueryTransformer(model_name=model_name)
    
    def determine_retrieval_strategy(self, query: str) -> Dict[str, Any]:
        """
        Dynamically determine the best retrieval strategy for a query
        
        Args:
            query: User query
            
        Returns:
            Retrieval strategy parameters
        """
        prompt = f"""
        Analyze the following query and determine the best retrieval strategy.
        
        Query: {query}
        
        Please specify:
        1. Top-k: How many documents to retrieve (1-10)
        2. Whether query expansion would help (yes/no)
        3. Whether the query contains multiple aspects that should be handled separately (yes/no)
        
        Format your response as a JSON object with fields "top_k", "expand_query", and "multi_aspect".
        """
        
        response = self.llm.predict(prompt)
        
        try:
            # Parse the response as JSON
            import json
            strategy = json.loads(response)
            return strategy
        except:
            # Fallback if parsing fails
            return {"top_k": 5, "expand_query": False, "multi_aspect": False}
    
    def retrieve_with_dynamic_strategy(self, query: str) -> List[Document]:
        """
        Retrieve documents using a dynamically determined strategy
        
        Args:
            query: User query
            
        Returns:
            Retrieved documents
        """
        # Determine strategy
        strategy = self.determine_retrieval_strategy(query)
        
        # Apply strategy
        if strategy.get("expand_query", False):
            query = self.query_transformer.expand_query(query)
        
        if strategy.get("multi_aspect", False):
            # Break down query into aspects
            sub_queries = self.query_transformer.generate_query_variations(query, num_variations=3)
            
            # Retrieve for each aspect
            all_docs = []
            for sub_q in sub_queries:
                docs = self.vector_store.similarity_search(sub_q, k=max(2, strategy.get("top_k", 5) // len(sub_queries)))
                all_docs.extend(docs)
            
            # Remove duplicates (simplified approach)
            unique_docs = []
            seen_content = set()
            for doc in all_docs:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    unique_docs.append(doc)
            
            return unique_docs[:strategy.get("top_k", 5)]
        else:
            # Standard retrieval
            return self.vector_store.similarity_search(query, k=strategy.get("top_k", 5))
```

### 3. Re-ranking Retrieved Documents

```python
class ReRanker:
    def __init__(self, model_name="gpt-4"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
    
    def cross_encoder_rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Re-rank documents using a cross-encoder approach with LLM
        
        Args:
            query: User query
            docs: Initially retrieved documents
            
        Returns:
            Re-ranked list of documents
        """
        scored_docs = []
        
        for doc in docs:
            # Create relevance scoring prompt
            prompt = f"""
            On a scale of 0 to 10, score how relevant the following document is to the query.
            Consider both topical relevance and how directly it addresses the information need.
            
            Query: {query}
            
            Document: {doc.page_content}
            
            Score (0-10):
            """
            
            # Get score
            response = self.llm.predict(prompt)
            
            try:
                score = float(response.strip())
            except:
                score = 5.0  # Default score if parsing fails
            
            scored_docs.append((doc, score))
        
        # Sort by score, descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return re-ranked documents
        return [doc for doc, _ in scored_docs]
    
    def reciprocal_rank_fusion(self, query: str, multiple_rankings: List[List[Document]]) -> List[Document]:
        """
        Combine multiple document rankings using Reciprocal Rank Fusion
        
        Args:
            query: User query
            multiple_rankings: List of document rankings from different retrievers
            
        Returns:
            Combined and re-ranked document list
        """
        # Track document scores
        doc_scores = {}
        
        # Constant from the RRF algorithm
        k = 60
        
        # Process each ranking
        for ranking in multiple_rankings:
            for rank, doc in enumerate(ranking):
                # Unique ID for the document
                doc_id = hash(doc.page_content)
                
                # Calculate RRF score: 1/(rank + k)
                score = 1.0 / (rank + k)
                
                # Add to document's total score
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {"doc": doc, "score": 0}
                
                doc_scores[doc_id]["score"] += score
        
        # Sort documents by final score
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        
        # Return just the documents in the new order
        return [item["doc"] for item in sorted_docs]
```

## Challenges and Future Directions

### Current Challenges

1. **Source reliability**: Ensuring retrieved information comes from trustworthy sources
2. **Context length limitations**: Managing the amount of retrieved information that can fit in the LLM's context window
3. **Evaluation complexity**: Developing robust metrics for RAG performance
4. **Temporal reasoning**: Handling time-dependent information correctly

### Future Directions

1. **Multimodal RAG**: Retrieving and reasoning over text, images, audio, and video
2. **Continuous learning**: Systems that update their knowledge base in real-time
3. **Personalized retrieval**: Tailoring retrieval to user preferences and history
4. **Hierarchical retrieval**: Using different granularities of information for different query types

## Practical Implementation Tips

1. **Start simple**: Begin with basic RAG and gradually add complexity
2. **Chunk wisely**: Document splitting significantly impacts retrieval quality
3. **Test thoroughly**: Evaluate your system against diverse queries
4. **Monitor quality**: Track metrics like retrieval precision and hallucination rates
5. **Consider hybrid retrieval**: Combine semantic search with keyword search for better results

RAG systems represent a significant advancement in making LLMs more factual, up-to-date, and reliable. By following the principles and techniques outlined in this document, you can build effective systems that deliver more accurate and trustworthy AI-generated content.