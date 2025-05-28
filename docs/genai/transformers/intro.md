---
title: Introduction to Transformers
sidebar_position: 1
description: Understanding the transformer architecture that powers modern generative AI
---

# Introduction to Transformers

The transformer architecture, introduced in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. in 2017, revolutionized natural language processing and laid the foundation for modern generative AI systems. Unlike previous sequence models that relied on recurrent or convolutional layers, transformers use self-attention mechanisms to process entire sequences in parallel, enabling more efficient training and better handling of long-range dependencies.

## Why Transformers Matter

Before transformers, sequence processing relied heavily on RNNs, LSTMs, and GRUs, which processed data sequentially and struggled with:

1. **Parallelization**: Sequential processing limited training speed
2. **Long-range dependencies**: Difficulty capturing relationships between distant elements
3. **Vanishing/exploding gradients**: Training instability with long sequences

Transformers addressed these challenges by:

1. **Enabling parallel processing**: All positions are processed simultaneously  
2. **Directly modeling all pairwise relationships**: Through attention mechanisms
3. **Creating more direct gradient paths**: Improving learning for long sequences

This architecture has become the backbone of nearly all state-of-the-art language models and many other generative AI systems.

## The Basic Transformer Design

The original transformer architecture consists of an encoder and a decoder, each containing multiple identical layers. Here's a simplified implementation of the key components:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = F.relu
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention block
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # Residual connection
        src = self.norm1(src)  # Layer normalization
        
        # Feed-forward block
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)  # Residual connection
        src = self.norm2(src)  # Layer normalization
        
        return src
```

## Key Components of Transformers

### 1. Positional Encoding

Since transformers process all positions at once, they need a way to understand token positions in a sequence:

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input embeddings [batch_size, seq_len, embedding_dim]
        Returns:
            Embeddings + positional encodings
        """
        x = x + self.pe[:, :x.size(1), :]
        return x
```

This creates a unique pattern for each position, allowing the model to understand sequence order without sacrificing parallelism.

### 2. Self-Attention Mechanism

The heart of transformers is the self-attention mechanism, which allows each position to attend to all positions in the sequence:

```python
def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Compute scaled dot-product attention
    
    Args:
        query: Query vectors [batch_size, num_heads, seq_len, d_k]
        key: Key vectors [batch_size, num_heads, seq_len, d_k]
        value: Value vectors [batch_size, num_heads, seq_len, d_v]
        mask: Optional mask [batch_size, 1, 1, seq_len]
    
    Returns:
        Context vectors and attention weights
    """
    d_k = query.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided (e.g., for padding or causal attention)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Compute context vectors
    context = torch.matmul(attention_weights, value)
    
    return context, attention_weights
```

### 3. Multi-Head Attention

Rather than using a single attention function, transformers use multiple attention heads to capture different types of relationships:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections and reshape for multi-head attention
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply scaled dot-product attention
        context, attn = scaled_dot_product_attention(q, k, v, mask)
        
        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out(context)
        
        return output, attn
```

### 4. Feed-Forward Networks

Each transformer layer includes a position-wise feed-forward network applied to each position separately:

```python
class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input [batch_size, seq_len, d_model]
        Returns:
            Output [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
```

### 5. Layer Normalization and Residual Connections

These components help stabilize training in deep transformer networks:

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Attention block with residual connection and layer norm
        attn_output, _ = self.attn(x, x, x, mask)
        x = x + self.dropout(attn_output)  # Residual connection
        x = self.norm1(x)  # Layer normalization
        
        # Feed-forward block with residual connection and layer norm
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)  # Residual connection
        x = self.norm2(x)  # Layer normalization
        
        return x
```

## Types of Transformer Architectures

### Encoder-Only Architectures

Models like BERT focus on understanding text by using bidirectional attention:

```python
class EncoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, 
                 d_ff=2048, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Embedding and positional encoding
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Process through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
            
        return x  # Returns contextual embeddings
```

### Decoder-Only Architectures

Models like GPT generate text autoregressively by using causal attention (can only see previous tokens):

```python
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, 
                 d_ff=2048, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Stack of decoder layers (but without cross-attention)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # Create causal mask (lower triangular)
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(x.device)
        
        # Embedding and positional encoding
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Process through decoder layers with causal masking
        for layer in self.layers:
            x = layer(x, mask=~causal_mask)
            
        # Project to vocabulary
        x = self.output_linear(x)
        return x
```

### Encoder-Decoder Architectures

The original transformer design, used in models like T5 and BART:

```python
class EncoderDecoderTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6,
                 d_ff=2048, max_seq_len=512, dropout=0.1):
        super().__init__()
        
        # Encoder
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.encoder_pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.decoder_pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_decoder_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        
    def encode(self, src, src_mask=None):
        x = self.encoder_embedding(src)
        x = self.encoder_pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.encoder_layers:
            x = layer(x, mask=src_mask)
            
        return x
        
    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Create causal mask for decoder
        seq_len = tgt.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(tgt.device)
        
        if tgt_mask is None:
            tgt_mask = ~causal_mask
            
        x = self.decoder_embedding(tgt)
        x = self.decoder_pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
            
        return x
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encode(src, src_mask)
        decoded = self.decode(tgt, memory, tgt_mask, memory_mask)
        output = self.output_linear(decoded)
        return output
```

## Modern Transformer Innovations

Since 2017, the transformer architecture has evolved significantly:

### 1. Efficient Attention Mechanisms

Traditional attention has O(n²) complexity with sequence length, leading to innovations like:

- **Sparse Attention**: Only attend to a subset of positions
- **Linear Attention**: Reduce complexity to O(n) using kernel methods
- **Local Attention**: Focus on neighboring tokens
- **FlashAttention**: Optimized implementation for GPU memory hierarchy

```python
# Example of simplified local attention
def local_attention(query, key, value, window_size=16):
    batch_size, num_heads, seq_len, d_k = query.size()
    
    # Initialize output tensor
    output = torch.zeros_like(value)
    
    # Process each position with local attention
    for i in range(seq_len):
        # Define local window boundaries
        start_idx = max(0, i - window_size // 2)
        end_idx = min(seq_len, i + window_size // 2 + 1)
        
        # Extract local keys and values
        local_keys = key[:, :, start_idx:end_idx, :]
        local_values = value[:, :, start_idx:end_idx, :]
        
        # Current query
        q = query[:, :, i:i+1, :]
        
        # Compute attention scores
        scores = torch.matmul(q, local_keys.transpose(-2, -1)) / math.sqrt(d_k)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Compute weighted sum
        output[:, :, i:i+1, :] = torch.matmul(attention_weights, local_values)
    
    return output
```

### 2. Parameter-Efficient Training

Methods to reduce the number of trainable parameters:

- **Adapter Layers**: Small bottleneck layers inserted into pre-trained models
- **LoRA (Low-Rank Adaptation)**: Decompose weight updates into low-rank matrices
- **Prefix Tuning**: Optimize only a small set of continuous prompts

```python
# Example of LoRA implementation
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=32):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank
        
        # Initialize with random weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        return self.scaling * torch.matmul(torch.matmul(x, self.lora_A), self.lora_B)
        
class LinearWithLoRA(nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=32):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(linear_layer.in_features, linear_layer.out_features, 
                              rank=rank, alpha=alpha)
        
    def forward(self, x):
        return self.linear(x) + self.lora(x)
```

### 3. Mixture of Experts (MoE)

Instead of processing all tokens through the entire model, MoE architectures use specialized sub-networks activated based on input:

```python
class SimpleExpert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))

class SparseMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([
            SimpleExpert(d_model, d_ff) for _ in range(num_experts)
        ])
        self.router = nn.Linear(d_model, num_experts)
        self.top_k = top_k
        self.num_experts = num_experts
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Reshape for routing
        x_flat = x.reshape(-1, d_model)  # [batch_size * seq_len, d_model]
        
        # Get routing probabilities
        router_logits = self.router(x_flat)  # [batch_size * seq_len, num_experts]
        
        # Select top-k experts
        router_probs = F.softmax(router_logits, dim=-1)
        values, indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize probabilities
        values = values / values.sum(dim=-1, keepdim=True)
        
        # Initialize output
        final_output = torch.zeros_like(x_flat)
        
        # Process input through selected experts
        for expert_idx in range(self.num_experts):
            # Find positions where this expert was selected
            expert_mask = (indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
                
            # Get inputs for this expert
            expert_inputs = x_flat[expert_mask]
            
            # Get outputs from expert
            expert_output = self.experts[expert_idx](expert_inputs)
            
            # Find corresponding weights for this expert
            expert_weights = torch.zeros(expert_mask.shape[0], device=x.device)
            for k in range(self.top_k):
                k_mask = indices[:, k] == expert_idx
                expert_weights[k_mask] = values[:, k][k_mask]
                
            # Apply weights and add to output
            expert_weights = expert_weights[expert_mask].unsqueeze(-1)
            final_output[expert_mask] += expert_output * expert_weights
            
        # Reshape back
        return final_output.reshape(batch_size, seq_len, d_model)
```

## Key Applications of Transformers

Transformers now power a wide range of AI applications:

1. **Natural Language Processing**
   - Language translation
   - Text summarization
   - Question answering
   - Text generation

2. **Computer Vision**
   - Image classification
   - Object detection
   - Image generation
   - Video understanding

3. **Speech Processing**
   - Speech recognition
   - Text-to-speech
   - Voice conversion

4. **Cross-modal Tasks**
   - Image captioning
   - Visual question answering
   - Text-to-image generation

## The Future of Transformer Architectures

Transformers continue to evolve in several key directions:

1. **Efficiency improvements**: Reducing computational and memory requirements
2. **Sparse models**: Activating only relevant parts of the network
3. **Longer context handling**: Processing books, conversations, and even videos
4. **Multimodal integration**: Unified architectures for text, images, audio, and video
5. **Structural priors**: Incorporating domain knowledge into architecture design

The transformer architecture has become the foundation of generative AI, enabling remarkable capabilities while continuing to evolve and improve. Understanding its core principles helps navigate the rapidly changing landscape of AI models and applications.