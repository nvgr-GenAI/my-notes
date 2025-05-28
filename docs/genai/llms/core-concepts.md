---
title: Core Concepts Behind LLMs
sidebar_position: 3
description: Detailed explanation of transformer architecture, attention mechanisms, feed-forward networks, and layer normalization
---

# Core Concepts Behind LLMs

To truly understand how large language models work, we need to explore the foundational architectural components that make them possible. This page dives into the key building blocks of modern LLMs, from the transformer architecture to the details that enable these models to understand and generate text.

## Transformer Architecture Overview

The transformer architecture, introduced in the 2017 paper "Attention Is All You Need," revolutionized natural language processing by eliminating the need for recurrent or convolutional layers while achieving superior performance through self-attention mechanisms.

### Key Innovations

The transformer introduced several key innovations over previous architectures:

1. **Parallelization**: Unlike RNNs, transformers process all tokens in parallel, dramatically speeding up training
2. **Self-attention**: Direct modeling of relationships between all tokens in a sequence
3. **Positional encoding**: Maintaining sequence order without recurrence
4. **Residual connections**: Facilitating training of very deep networks

### High-Level Structure

At a high level, the transformer consists of:

- **Encoder stack**: For understanding input (used in encoder-only and encoder-decoder models)
- **Decoder stack**: For generating output (used in decoder-only and encoder-decoder models)

```python
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, 
                 vocab_size,
                 d_model=512, 
                 nhead=8, 
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048, 
                 dropout=0.1,
                 activation="gelu"):
        super().__init__()
        
        # Token embedding + positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Encoder and decoder stacks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Final output projection
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embed and encode positions of source sequence
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Process through encoder
        memory = self.encoder(src, src_mask)
        
        # Embed and encode positions of target sequence
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        
        # Process through decoder
        output = self.decoder(tgt, memory, tgt_mask)
        
        # Project to vocabulary
        return self.output_layer(output)
```

### Modern Variations

Contemporary LLMs build upon the original transformer architecture with various modifications:

1. **GPT-style models**: Decoder-only transformers with masked self-attention
2. **PaLM/Gemini**: Dense attention patterns with parallel experts
3. **Mixture-of-Experts**: Models like Mixtral that activate only part of the network per token
4. **Multi-query attention**: Optimization for inference efficiency
5. **FlashAttention**: Algorithm improvement for more efficient computation

## Attention Mechanism Explained

The attention mechanism is the core innovation of transformer models, allowing the model to focus on different parts of the input sequence when producing each output token.

### Self-Attention Mathematics

In mathematical terms, self-attention computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where:
- $Q$ (query), $K$ (key), and $V$ (value) are learned linear projections of the input
- $d_k$ is the dimension of the key vectors, used for scaling
- $\text{softmax}$ normalizes the attention scores to sum to 1

Let's implement this in code:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # Linear projections
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear projections
        q = self.query(x)  # (batch_size, seq_len, d_model)
        k = self.key(x)    # (batch_size, seq_len, d_model)
        v = self.value(x)  # (batch_size, seq_len, d_model)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_model)  # (batch_size, seq_len, seq_len)
        
        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)
        
        # Weighted sum of values
        output = torch.matmul(attn_weights, v)  # (batch_size, seq_len, d_model)
        
        # Final projection
        output = self.out(output)  # (batch_size, seq_len, d_model)
        
        return output, attn_weights
```

### Multi-Head Attention

Rather than performing a single attention function, transformers use multi-head attention, which:
1. Allows the model to jointly attend to information from different positions
2. Creates multiple "representation subspaces"
3. Improves model expressiveness

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out = nn.Linear(d_model, d_model)
        
    def split_heads(self, x, batch_size):
        # Split the last dimension into (num_heads, d_k)
        # and transpose to (batch_size, num_heads, seq_len, d_k)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections and split heads
        q = self.split_heads(self.query(q), batch_size)  # (batch, num_heads, seq_len, d_k)
        k = self.split_heads(self.key(k), batch_size)    # (batch, num_heads, seq_len, d_k)
        v = self.split_heads(self.value(v), batch_size)  # (batch, num_heads, seq_len, d_k)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, d_k)
        
        # Transpose and reshape: (batch, seq_len, num_heads, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.out(attn_output)
        
        return output, attn_weights
```

### Attention Masks

In decoder-only or encoder-decoder models, attention masks prevent tokens from attending to future positions:

```python
def create_causal_mask(size):
    """Create a causal mask for decoder self-attention"""
    mask = torch.ones(size, size)
    mask = torch.triu(mask, diagonal=1)  # Upper triangular part (excluding diagonal)
    return mask == 0  # Convert to boolean mask where 1s allow attention
    
# Example usage
seq_len = 10
mask = create_causal_mask(seq_len)

plt.figure(figsize=(6, 6))
plt.imshow(mask, cmap='binary')
plt.title('Causal Attention Mask')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.colorbar()
```

### Attention Patterns and Interpretation

Attention patterns can provide insights into what the model is focusing on:

1. **Diagonal patterns**: Attending to the current token
2. **Horizontal bands**: Global focus on important tokens
3. **Vertical bands**: Tokens that influence many other tokens
4. **Checkerboard patterns**: Often seen in positional attention

Researchers use attention visualization to better understand model behavior:

```python
def visualize_attention(tokens, attention_weights, layer=0, head=0):
    """Visualize attention weights between tokens"""
    attn = attention_weights[layer][head].detach().numpy()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(attn)
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.title(f"Attention weights (Layer {layer+1}, Head {head+1})")
    plt.colorbar()
```

### Efficient Attention Implementations

As context windows grow, attention's O(n²) complexity becomes problematic. Several methods address this:

1. **FlashAttention**: Memory-efficient algorithm reducing HBM access
   ```python
   # Conceptual FlashAttention (not actual implementation)
   def flash_attention(q, k, v, block_size=1024):
       """Memory-efficient attention using block-wise computations"""
       batch_size, seq_len, head_dim = q.shape
       output = torch.zeros_like(q)
       
       # Process in blocks to minimize memory transfers
       for i in range(0, seq_len, block_size):
           q_block = q[:, i:i+block_size]
           
           for j in range(0, seq_len, block_size):
               k_block = k[:, j:j+block_size]
               v_block = v[:, j:j+block_size]
               
               # Compute attention for this block
               scores = torch.matmul(q_block, k_block.transpose(-2, -1)) / math.sqrt(head_dim)
               if i < j + block_size and causal:
                   # Apply causal mask within relevant blocks
                   mask = torch.ones_like(scores, dtype=torch.bool)
                   mask = torch.triu(mask, diagonal=j-i+1)
                   scores.masked_fill_(mask, float("-inf"))
                   
               attn_weights = torch.softmax(scores, dim=-1)  # Partial softmax
               output[:, i:i+block_size] += torch.matmul(attn_weights, v_block)
               
       return output
   ```

2. **Sparse Attention**: Only compute attention for selected positions
   ```python
   def sparse_attention(q, k, v, sparsity_pattern):
       """Compute attention only for positions indicated in sparsity_pattern"""
       scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
       
       # Apply sparsity pattern (0s in the pattern block attention)
       scores.masked_fill_(~sparsity_pattern, float("-inf"))
       
       attn_weights = torch.softmax(scores, dim=-1)
       return torch.matmul(attn_weights, v)
   ```

3. **Linear Attention**: Approximations with lower computational complexity
   ```python
   def linear_attention(q, k, v):
       """O(n) attention approximation using kernel trick"""
       # Apply feature map φ (here using ELU + 1)
       q_prime = torch.nn.functional.elu(q) + 1
       k_prime = torch.nn.functional.elu(k) + 1
       
       # Compute using associative property: Attention(Q,K,V) ≈ φ(Q) · (φ(K)ᵀ · V) / (φ(Q) · sum(φ(K))ᵀ)
       kv = torch.matmul(k_prime.transpose(-2, -1), v)
       z = 1.0 / torch.matmul(q_prime, torch.sum(k_prime, dim=-2).unsqueeze(-1))
       return torch.matmul(q_prime, kv) * z
   ```

4. **Grouped-Query Attention (GQA)**: Sharing K/V projections across multiple heads
   ```python
   class GroupedQueryAttention(nn.Module):
       def __init__(self, d_model, num_q_heads=8, num_kv_heads=2):
           super().__init__()
           self.d_model = d_model
           self.num_q_heads = num_q_heads
           self.num_kv_heads = num_kv_heads
           self.head_dim = d_model // num_q_heads
           
           self.q_proj = nn.Linear(d_model, d_model)
           self.k_proj = nn.Linear(d_model, self.head_dim * num_kv_heads)
           self.v_proj = nn.Linear(d_model, self.head_dim * num_kv_heads)
           self.out_proj = nn.Linear(d_model, d_model)
           
       def forward(self, x, mask=None):
           batch_size, seq_len = x.shape[:2]
           
           # Project q, k, v
           q = self.q_proj(x).view(batch_size, seq_len, self.num_q_heads, self.head_dim)
           k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
           v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
           
           # Duplicate k,v to match number of q heads
           # If num_q_heads=8, num_kv_heads=2, each kv head serves 4 q heads
           k = torch.repeat_interleave(k, self.num_q_heads // self.num_kv_heads, dim=2)
           v = torch.repeat_interleave(v, self.num_q_heads // self.num_kv_heads, dim=2)
           
           # Transpose for attention calculation
           q = q.transpose(1, 2)  # (batch, num_q_heads, seq_len, head_dim)
           k = k.transpose(1, 2)  # (batch, num_q_heads, seq_len, head_dim)
           v = v.transpose(1, 2)  # (batch, num_q_heads, seq_len, head_dim)
           
           # Calculate attention
           scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
           if mask is not None:
               scores = scores.masked_fill(mask == 0, -1e9)
           
           attn_weights = F.softmax(scores, dim=-1)
           attn_output = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)
           
           # Transpose and reshape
           attn_output = attn_output.transpose(1, 2).contiguous()
           output = attn_output.view(batch_size, seq_len, -1)
           
           return self.out_proj(output)
   ```

## Feed-Forward Networks in Transformers

Between attention layers, transformers use position-wise feed-forward networks (FFNs), which are applied independently to each position.

### Standard Architecture

The standard FFN in transformers consists of:
1. Linear projection to a larger dimension
2. Non-linear activation (typically GELU in modern LLMs)
3. Linear projection back to model dimension

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation="gelu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
```

### Role in Transformers

FFNs serve several important functions:

1. **Increased capacity**: The expanded dimension (typically 4x the model dimension) gives the model more parameters to learn complex patterns
2. **Position-wise processing**: Each token is processed independently
3. **Non-linearity**: The activation function introduces non-linear transformations
4. **Feature transformation**: Converting between the attention subspace and the representation subspace

### Modern Variations

Contemporary LLMs use several variations of the standard FFN:

1. **Gated FFNs**: Adding gating mechanisms to control information flow
   ```python
   class GatedFeedForward(nn.Module):
       def __init__(self, d_model, d_ff, dropout=0.1):
           super().__init__()
           self.gate_proj = nn.Linear(d_model, d_ff)
           self.up_proj = nn.Linear(d_model, d_ff)
           self.down_proj = nn.Linear(d_ff, d_model)
           self.dropout = nn.Dropout(dropout)
           self.activation = nn.GELU()
           
       def forward(self, x):
           gate = self.activation(self.gate_proj(x))
           up = self.up_proj(x)
           x = gate * up  # Element-wise multiplication
           x = self.dropout(x)
           x = self.down_proj(x)
           return x
   ```

2. **GLU variants**: Using Gated Linear Units for improved performance
   ```python
   class GLUFeedForward(nn.Module):
       def __init__(self, d_model, d_ff, dropout=0.1):
           super().__init__()
           self.w1 = nn.Linear(d_model, d_ff)
           self.w2 = nn.Linear(d_model, d_ff)
           self.w3 = nn.Linear(d_ff, d_model)
           self.dropout = nn.Dropout(dropout)
           self.activation = nn.GELU()
           
       def forward(self, x):
           gating = self.activation(self.w1(x))
           values = self.w2(x)
           x = gating * values
           x = self.dropout(x)
           x = self.w3(x)
           return x
   ```

3. **Expert FFNs**: Using Mixture of Experts (MoE) for greater parameter efficiency
   ```python
   class MoEFeedForward(nn.Module):
       def __init__(self, d_model, d_ff, num_experts=8, top_k=2):
           super().__init__()
           self.d_model = d_model
           self.d_ff = d_ff
           self.num_experts = num_experts
           self.top_k = top_k
           
           # Router for selecting experts
           self.router = nn.Linear(d_model, num_experts)
           
           # Create multiple expert FFNs
           self.experts = nn.ModuleList([
               FeedForward(d_model, d_ff)
               for _ in range(num_experts)
           ])
           
       def forward(self, x):
           batch_size, seq_len, d_model = x.shape
           
           # Get router scores
           router_logits = self.router(x)  # (batch, seq_len, num_experts)
           
           # Select top-k experts per token
           router_probs = F.softmax(router_logits, dim=-1)
           top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
           
           # Normalize probabilities
           top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
           
           # Initialize expert outputs
           final_output = torch.zeros(batch_size, seq_len, d_model, device=x.device)
           
           # Run selected experts
           for i in range(self.top_k):
               # Extract expert indices for this position
               expert_idx = top_k_indices[:, :, i]  # (batch, seq_len)
               expert_prob = top_k_probs[:, :, i].unsqueeze(-1)  # (batch, seq_len, 1)
               
               # Process tokens through their assigned experts
               for j in range(self.num_experts):
                   # Find tokens routed to expert j
                   mask = (expert_idx == j).unsqueeze(-1)  # (batch, seq_len, 1)
                   
                   if not mask.any():
                       continue
                   
                   # Process through this expert
                   expert_output = self.experts[j](x)
                   
                   # Scale by router probability and add to output
                   final_output += mask * expert_output * expert_prob
                   
           return final_output
   ```

### FFN Size and Performance

The size of feed-forward networks significantly impacts model performance:

1. **Parameter efficiency**: FFNs typically contain 2/3 of a model's parameters
2. **Scaling laws**: Performance scales with FFN dimensions, but with diminishing returns
3. **Computation cost**: Larger FFNs increase both training and inference costs

## Layer Normalization and Residual Connections

Transformers rely heavily on two techniques to stabilize training and improve gradient flow: layer normalization and residual connections.

### Layer Normalization

Layer normalization normalizes activations across feature dimensions for each example independently, helping to:
1. Stabilize training
2. Reduce internal covariate shift
3. Allow for larger learning rates

```python
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(features))
        self.shift = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.scale * (x - mean) / (std + self.eps) + self.shift
```

### Variants of Normalization

Modern LLMs use several normalization variants:

1. **RMSNorm**: Simplified version that only normalizes by root mean square
   ```python
   class RMSNorm(nn.Module):
       def __init__(self, features, eps=1e-6):
           super().__init__()
           self.scale = nn.Parameter(torch.ones(features))
           self.eps = eps
           
       def forward(self, x):
           # Only use RMS, no mean subtraction
           rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
           return self.scale * x / rms
   ```

2. **Pre-LayerNorm vs Post-LayerNorm**: Different positions in the architecture
   ```python
   # Pre-LN transformer block (more stable training)
   def pre_ln_block(x, self_attn, ff, norm1, norm2):
       # Normalize before attention
       x1 = norm1(x)
       attn_output = self_attn(x1, x1, x1)
       x = x + attn_output  # Residual connection
       
       # Normalize before feed-forward
       x2 = norm2(x)
       ff_output = ff(x2)
       x = x + ff_output  # Residual connection
       
       return x
       
   # Post-LN transformer block (better performance but harder to train)
   def post_ln_block(x, self_attn, ff, norm1, norm2):
       # Attention with residual connection, then normalize
       attn_output = self_attn(x, x, x)
       x = norm1(x + attn_output)
       
       # Feed-forward with residual connection, then normalize
       ff_output = ff(x)
       x = norm2(x + ff_output)
       
       return x
   ```

### Residual Connections

Residual connections (or skip connections) directly add input to output, helping with:
1. Gradient flow during backpropagation
2. Training of very deep networks
3. Creating ensemble-like effects

```python
class ResidualConnection(nn.Module):
    def __init__(self, sublayer, d_model, dropout=0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, *args):
        # Apply residual connection with normalization and dropout
        return x + self.dropout(self.sublayer(self.norm(x), *args))
```

### Impact on Deep Transformer Training

The combination of layer normalization and residual connections is essential for training deep transformers:

1. **Gradient propagation**: Residual connections provide direct paths for gradients
2. **Signal preservation**: Both techniques help preserve signal magnitude through deep networks
3. **Initialization sensitivity**: Proper initialization becomes less critical
4. **Optimization stability**: Reducing variance in activations helps stabilization

### Full Transformer Block

A complete transformer encoder block combining all these elements:

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, pre_norm=True):
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.attn_norm = LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.ff_norm = LayerNorm(d_model)
        self.ff_dropout = nn.Dropout(dropout)
        
        # Whether to use pre-norm or post-norm
        self.pre_norm = pre_norm
        
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        if self.pre_norm:
            # Pre-norm architecture
            x_norm = self.attn_norm(x)
            attn_output, _ = self.attention(x_norm, x_norm, x_norm, mask)
            x = x + self.attn_dropout(attn_output)
            
            # Feed-forward with residual connection
            x_norm = self.ff_norm(x)
            ff_output = self.feed_forward(x_norm)
            x = x + self.ff_dropout(ff_output)
        else:
            # Post-norm architecture
            attn_output, _ = self.attention(x, x, x, mask)
            x = self.attn_norm(x + self.attn_dropout(attn_output))
            
            # Feed-forward with residual connection
            ff_output = self.feed_forward(x)
            x = self.ff_norm(x + self.ff_dropout(ff_output))
        
        return x
```

## Putting It All Together: Transformer Decoder for LLMs

Most modern LLMs are based on transformer decoder architectures. Here's a simplified implementation of a GPT-style decoder:

```python
class GPTDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=12, num_heads=12, 
                 d_ff=3072, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embedding (learned)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout, pre_norm=True)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = LayerNorm(d_model)
        
        # Output projection
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize parameters
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Initialize linear layers with small random values
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, input_ids):
        # Get sequence length and create attention mask
        seq_length = input_ids.size(1)
        attention_mask = torch.tril(torch.ones((seq_length, seq_length), device=input_ids.device))
        
        # Get embeddings
        token_embeddings = self.token_embedding(input_ids)  # (batch, seq_len, d_model)
        position_embeddings = self.pos_embedding[:, :seq_length, :]  # (1, seq_len, d_model)
        
        # Combine embeddings
        x = self.dropout(token_embeddings + position_embeddings)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, mask=attention_mask)
            
        # Final normalization
        x = self.norm(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=0):
        self.eval()
        
        # Start with the provided input_ids
        generated = input_ids.clone()
        
        # Generate one token at a time
        for _ in range(max_new_tokens):
            # Use only the most recent context to avoid out-of-memory
            input_chunk = generated[:, -min(generated.size(1), 1024):]
            
            # Get predictions for next token
            with torch.no_grad():
                outputs = self.forward(input_chunk)
                next_token_logits = outputs[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / max(temperature, 1e-7)
            
            # Apply top-k sampling
            if top_k > 0:
                # Set all logits outside top-k to -inf
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the probability distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to the generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
        return generated
```

## Conclusion

The fundamental architecture of modern LLMs builds upon the transformer's self-attention mechanism, feed-forward networks, and normalization techniques. These core components, when scaled up and optimized, enable the remarkable capabilities we see in today's language models.

As research progresses, many of these components continue to evolve with innovations in efficiency, performance, and scaling properties. Understanding these building blocks provides a solid foundation for working with and extending LLM technology.

In the next section, we'll explore how LLMs are trained and what makes training these massive models both challenging and effective.