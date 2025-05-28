---
title: Introduction to Large Language Models
sidebar_position: 1
description: Understanding Large Language Models (LLMs) and their foundational concepts
---

# Introduction to Large Language Models

Large Language Models (LLMs) are a class of generative AI systems that have transformed natural language processing. They can generate human-like text, translate languages, write different kinds of creative content, and answer questions in an informative way.

## What Are Large Language Models?

LLMs are neural networks trained on vast amounts of text data to predict the next word or token in a sequence. Through this pre-training objective, they develop capabilities far beyond simple prediction, including:

- Understanding context and semantics
- Following complex instructions
- Reasoning through problems
- Generating creative content
- Answering questions based on learned knowledge

## How LLMs Work

At their core, LLMs operate through several key mechanisms:

### Tokenization

Before processing text, LLMs convert it into tokens:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "Large language models use tokenizers to convert text into numbers."

# Convert text to tokens
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")

# Convert tokens back to text
decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")

# Visualize each token
for token in tokens:
    print(f"Token {token}: '{tokenizer.decode([token])}'")
```

Different models use different tokenization approaches:
- **Word-based**: Each token is a full word
- **Subword-based**: Common words are single tokens, uncommon words are split into multiple tokens
- **Character-based**: Each character is a separate token
- **Byte-Pair Encoding (BPE)**: Used in many modern LLMs, merges common character pairs iteratively

#### Why Tokenization Matters

Tokenization affects model performance and behavior in several ways:
- **Vocabulary efficiency**: Balancing coverage vs memory usage
- **Out-of-vocabulary handling**: How models deal with unfamiliar words
- **Multilingual capabilities**: Some tokenizers work better across languages
- **Token economy**: Users are charged per token in commercial APIs

For example, GPT models using BPE might tokenize "indivisible" as ["in", "divis", "ible"], while a character-based tokenizer would use 11 separate tokens.

### Model Architecture: Transformers

LLMs are built on the transformer architecture, which has several key components:

#### Embedding Layer

Transforms tokens into high-dimensional vectors that capture semantic meaning:

```python
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, x):
        return self.embedding(x)
```

These embeddings encode semantic relationships, allowing models to understand that words like "king" and "queen" or "cat" and "feline" are related conceptually.

#### Positional Encoding

Since transformers process all tokens simultaneously (not sequentially), they need position information:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def get_positional_encoding(seq_len, d_model):
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    
    return pos_encoding

# Visualize positional encodings
pos_encoding = get_positional_encoding(100, 128)

plt.figure(figsize=(10, 6))
plt.imshow(pos_encoding, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Positional Encodings')
plt.xlabel('Embedding Dimension')
plt.ylabel('Token Position')
```

Modern LLMs may use more advanced positional encoding schemes:
- **Learned positional embeddings**: Trained rather than fixed
- **Rotary Position Embedding (RoPE)**: Encodes position through rotation in vector space
- **ALiBi**: Attention with Linear Biases for better extrapolation to longer sequences

### Attention Mechanisms

The transformer architecture uses attention mechanisms to weigh the importance of different tokens when generating each output token:

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Simplified self-attention implementation
def self_attention(query, key, value):
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(key.shape[-1])
    
    # Apply softmax to get attention weights
    attention_weights = torch.softmax(scores, dim=-1)
    
    # Compute output
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights

# Example usage
seq_len = 5
d_model = 64
batch_size = 1

# Create random query, key, value matrices
query = torch.rand(batch_size, seq_len, d_model)
key = torch.rand(batch_size, seq_len, d_model)
value = torch.rand(batch_size, seq_len, d_model)

output, attention_weights = self_attention(query, key, value)

# Visualizing attention weights
plt.figure(figsize=(8, 6))
plt.imshow(attention_weights[0].detach().numpy(), cmap='viridis')
plt.title('Self-Attention Weights')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.colorbar(label='Attention Weight')
```

#### Types of Attention

Modern LLMs use variations of attention for efficiency and performance:

- **Multi-head attention**: Multiple attention operations in parallel
- **Flash attention**: Memory-efficient algorithm for faster computation
- **Grouped-query attention (GQA)**: Sharing key and value projections across groups of query heads
- **Multi-query attention (MQA)**: Using a single key-value head for all query heads
- **Sliding window attention**: Focusing on local context to handle long sequences

The attention formula is:

<!-- $$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}}) \cdot V$$ -->

<!-- Where $Q$ (query), $K$ (key), and $V$ (value) are learned projections of the input embeddings. -->

### Feed-Forward Networks

Between attention layers, LLMs use position-wise feed-forward networks:

```python
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # Modern LLMs often use GELU
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
```

These networks:
- Process each position independently
- Are often much wider than the model dimension (4-8x)
- Function as mini-MLPs that learn complex patterns
- Account for most of the model parameters

### Layer Normalization

LLMs use normalization to stabilize training:

```python
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias
```

Modern models use variations like RMSNorm (Root Mean Square Normalization) for improved efficiency.

### Training Process

LLMs learn through multiple stages:

1. **Pre-training**:
   - Massive text corpora (hundreds of billions to trillions of tokens)
   - Next-token prediction objective
   - Training on thousands of GPUs/TPUs for weeks or months

2. **Fine-tuning**:
   - Supervised fine-tuning (SFT) on instruction-following examples
   - Reinforcement Learning from Human Feedback (RLHF)
   - Direct Preference Optimization (DPO)

Pre-training uses a simple loss function for next-token prediction:

```python
def compute_loss(logits, targets):
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
```

### Context Window

The context window defines how much text an LLM can process and generate in a single run:

- Early LLMs: 512-2048 tokens
- Modern LLMs (2023): 8K-32K tokens
- Advanced LLMs (2024-2025): 128K-1M+ tokens

Longer context windows enable:
- Processing entire documents
- Maintaining coherent conversations over many turns
- Analyzing and summarizing lengthy content

However, they also introduce challenges:
- Increased computational requirements (attention is O(n²) with sequence length)
- Memory usage scaling
- Attention dilution over very long contexts
- Training data scarcity for long-context examples

#### Context Window Techniques

Several approaches help models manage longer contexts:

- **Sparse attention patterns**: Processing selected tokens instead of all-to-all attention
- **Hierarchical approaches**: Processing at different levels of granularity
- **Retrieval augmentation**: Fetching relevant context as needed
- **Sliding window approaches**: Moving a fixed-size window through long documents

### Inference and Generation

During generation, LLMs use sampling strategies to produce diverse, high-quality text:

```python
def generate_text(model, tokenizer, prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate with temperature and top-p sampling
    output = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,  # Controls randomness (lower is more deterministic)
        top_p=0.9,        # Nucleus sampling (consider tokens with cumulative probability p)
        top_k=50,         # Limit to top k tokens
        no_repeat_ngram_size=2  # Avoid repeating n-grams
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)
```

#### Decoding Strategies

Different decoding methods balance quality, diversity, and coherence:

- **Greedy decoding**: Always select the most probable token
- **Beam search**: Maintain multiple candidate sequences
- **Top-k sampling**: Sample from the k most likely tokens
- **Top-p (nucleus) sampling**: Sample from tokens comprising probability mass p
- **Temperature sampling**: Control randomness via temperature parameter

### Emergent Abilities

As LLMs scale in size and training data, they demonstrate emergent abilities that weren't explicitly trained for:

1. **In-context learning**: Using examples in the prompt to learn new tasks
2. **Chain-of-thought reasoning**: Breaking complex problems into steps
3. **Instruction following**: Adapting to diverse user instructions
4. **Tool use**: Using external tools based on natural language descriptions
5. **Multimodal understanding**: Processing images and text together (in newer models)

These capabilities often appear as model scale crosses certain thresholds, suggesting qualitative changes in model behavior rather than just incremental improvements.

### Parameter Efficient Fine-Tuning (PEFT)

Instead of updating all model weights, PEFT techniques adapt only a small subset:

```python
# Example using LoRA (Low-Rank Adaptation)
from peft import get_peft_model, LoraConfig, TaskType

# Define LoRA configuration
lora_config = LoraConfig(
    r=16,                     # Rank of update matrices
    lora_alpha=32,            # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which modules to adapt
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA to model
model = get_peft_model(base_model, lora_config)

# Now only LoRA parameters will be trained (typically <1% of full model)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

Popular PEFT methods include:
- **LoRA/QLoRA**: Low-rank adaptations of weight matrices
- **Adapters**: Small bottleneck networks inserted between layers
- **Prompt tuning**: Tuning continuous embeddings prepended to inputs
- **(IA)³**: Injecting trainable vectors into attention and feed-forward computations

## LLM Architectures

Modern LLMs are based on variations of the transformer architecture:

### Decoder-Only Models

Used in most general-purpose LLMs (GPT family, LLaMA, Claude):

- Process text from left to right (unidirectional)
- Each token can only attend to previous tokens
- Well-suited for text generation tasks
- Examples: GPT-4, Claude, LLaMA, Mistral, Falcon

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a decoder-only model (GPT-2)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Generate text
input_text = "The future of artificial intelligence will"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(
    input_ids, 
    max_length=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

### Encoder-Only Models

Focused on understanding text (BERT family):

- Bidirectional attention (each token can see all other tokens)
- Better for comprehension and classification tasks
- Not designed for text generation
- Examples: BERT, RoBERTa, DeBERTa

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

# Load an encoder-only model (BERT)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# Create a masked sentence
text = "The capital of France is [MASK]."
inputs = tokenizer(text, return_tensors="pt")

# Get token predictions
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

# Find the predicted token for the masked position
mask_token_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]
predicted_token_id = predictions[0, mask_token_index].argmax(axis=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"Predicted word: {predicted_token}")
```

### Encoder-Decoder Models

Designed for sequence-to-sequence tasks:

- Encoder processes input sequence
- Decoder generates output sequence
- Well-suited for translation, summarization
- Examples: T5, BART, mT5

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load an encoder-decoder model (T5)
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Example: translation
input_text = "translate English to German: The house is beautiful."
input_ids = tokenizer.encode(input_text, return_tensors="pt")

outputs = model.generate(input_ids, max_length=40)
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Translation: {translated_text}")
```

## LLMs in Production

Deploying LLMs in production involves several considerations:

### API Access vs. Self-hosting

**API Access:**
- Lower technical barrier
- Managed infrastructure
- Regular updates and improvements
- Pay-per-use pricing
- Examples: OpenAI API, Anthropic API, Google Gemini API

**Self-hosting:**
- Full control over model and data
- No data privacy concerns
- No ongoing API costs
- Higher technical requirements
- Examples: LLaMA, Mistral, Falcon

### Quantization and Optimization

Reducing model size for efficient deployment:

```python
# Example of loading a quantized model with bitsandbytes
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 4-bit quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    quantization_config=quantization_config
)

# Model is now loaded in 4-bit precision, requiring significantly less memory
```

### Latency and Response Time

Various strategies can reduce response time:
- **Speculative decoding**: Generate multiple tokens at once
- **KV caching**: Store key-value pairs to avoid recomputation
- **Batching**: Process multiple requests together
- **Model distillation**: Create smaller, faster models that approximate larger ones

### Responsible Use

LLMs present unique challenges:
- **Hallucinations**: False or misleading information
- **Bias**: Reflecting or amplifying societal biases
- **Privacy**: Handling sensitive user data
- **Misuse**: Potential for generating harmful content

## The Future of LLMs

The landscape of LLMs continues to evolve rapidly in several directions:

1. **Multimodal capabilities**: Integrating vision, audio, and text understanding
2. **Specialized domain models**: Fine-tuned for medicine, law, science, and other fields
3. **Improved reasoning**: Enhanced logical and mathematical capabilities
4. **Smaller, more efficient models**: Maintaining capabilities with reduced compute
5. **Longer context windows**: Processing and maintaining information over extended contexts
6. **Agentic behaviors**: Using tools and executing multi-step tasks autonomously

LLMs have fundamentally changed how we interact with AI systems, and they continue to advance at a remarkable pace, opening new possibilities for augmenting human capabilities and automating complex tasks.

## Key Milestones in LLM Development

| Year | Model | Organization | Parameters | Major Advancement |
|------|-------|--------------|------------|-------------------|
| 2018 | GPT-1 | OpenAI | 117M | First GPT model using transformers |
| 2018 | BERT | Google | 340M | Bidirectional context for understanding |
| 2019 | GPT-2 | OpenAI | 1.5B | Impressive text generation capabilities |
| 2020 | GPT-3 | OpenAI | 175B | Few-shot learning through in-context examples |
| 2021 | LaMDA | Google | 137B | Focus on dialog applications |
| 2022 | PaLM | Google | 540B | Pathways architecture for efficient training |
| 2022 | ChatGPT | OpenAI | ~175B | Conversational interface with RLHF |
| 2023 | GPT-4 | OpenAI | >1T estimated | Multimodal capabilities, improved reasoning |
| 2023 | LLaMA | Meta | 7B-70B | Open-weights models with competitive performance |
| 2023 | Claude | Anthropic | Not disclosed | Constitutional AI approach |
| 2023 | Mixtral 8x7B | Mistral AI | 45B effective | Sparse Mixture of Experts architecture |
| 2024 | Claude 3 | Anthropic | Not disclosed | Improved reasoning and multimodal abilities |
| 2024 | GPT-4o | OpenAI | Not disclosed | Optimized for multimodal performance |
| 2025 | Gemini Ultra 2 | Google | Not disclosed | Advanced reasoning and planning capabilities |