---
title: Transformer Architecture
sidebar_position: 1
description: A deep dive into the architecture of transformers, the backbone of modern language models
---

# Transformer Architecture

The transformer architecture, introduced in the groundbreaking 2017 paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al., revolutionized natural language processing and forms the foundation of modern large language models (LLMs).

## Why Transformers Matter

Before transformers, sequence modeling relied primarily on recurrent neural networks (RNNs) like LSTMs and GRUs. While effective, these architectures had limitations:

- **Sequential computation**: Processing tokens one after another, limiting parallelization
- **Limited context window**: Difficulty maintaining context over long sequences  
- **Vanishing gradients**: Information loss over long sequences

Transformers addressed these challenges through a novel architecture based on self-attention mechanisms, enabling:

- **Parallel processing**: Computing representations for all tokens simultaneously
- **Global context**: Each token can directly attend to all other tokens
- **Stable gradient flow**: Better preservation of information across the network

## High-Level Architecture

At a high level, the transformer consists of:

1. **Encoder**: Processes the input sequence to create contextualized representations
2. **Decoder**: Generates output tokens based on the encoder representations and previous outputs

However, many modern LLMs use decoder-only or encoder-only architectures:

- **Decoder-only** (GPT family, LLaMA): Specialized in text generation
- **Encoder-only** (BERT, RoBERTa): Specialized in understanding and representation
- **Encoder-decoder** (T5, BART): Used for sequence-to-sequence tasks like translation

![Transformer Architecture](/img/docusaurus.png)

## Core Components

### 1. Input Embedding

Text is first tokenized and converted into numerical vectors:

```python
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        
    def forward(self, x):
        return self.embedding(x) * (self.embedding_dim ** 0.5)  # Scaling factor
```

### 2. Positional Encoding

Since transformers process all tokens in parallel, positional information is added to maintain sequence order:

```python
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_length=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                            (-math.log(10000.0) / embedding_dim))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input embeddings
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

### 3. Multi-Head Self-Attention

The core innovation of transformers, allowing tokens to attend to other tokens:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(embedding_dim, embedding_dim)
        self.k_linear = nn.Linear(embedding_dim, embedding_dim)
        self.v_linear = nn.Linear(embedding_dim, embedding_dim)
        self.out_linear = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear projections and reshape for multi-head attention
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, T, D/H]
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Calculate attention scores
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
            
        # Apply softmax to get attention weights
        attention = self.dropout(torch.softmax(energy, dim=-1))
        
        # Apply attention to values
        x = torch.matmul(attention, V)  # [B, H, T, D/H]
        
        # Reshape and apply output projection
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.embedding_dim)
        output = self.out_linear(x)
        
        return output, attention
```

Multi-head attention is the mechanism that enables transformers to capture different relationships between tokens by:

1. Projecting the input into multiple sets of queries (Q), keys (K), and values (V)
2. Computing separate attention patterns for each head
3. Combining the results from all heads

The attention computation follows:

<!-- $$
Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \cdot V
$$ -->

<!-- Where $d_k$ is the dimension of the key vectors. -->

### 4. Feed-Forward Network

After attention, each token's representation is processed by a feed-forward network:

```python
class FeedForward(nn.Module):
    def __init__(self, embedding_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### 5. Layer Normalization

Applied before each major component to stabilize training:

```python
class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        norm_x = (x - mean) / (std + self.eps)
        return self.gamma * norm_x + self.beta
```

### 6. Residual Connections

Used throughout to help with gradient flow:

```python
class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        # Components
        self.self_attn = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.self_attn_norm = LayerNorm(embedding_dim)
        self.ff = FeedForward(embedding_dim, ff_dim, dropout)
        self.ff_norm = LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention block with residual connection and normalization
        _x = self.self_attn_norm(x)
        _x, attention = self.self_attn(_x, _x, _x, mask)
        x = x + self.dropout(_x)  # Residual connection
        
        # Feed-forward block with residual connection and normalization
        _x = self.ff_norm(x)
        _x = self.ff(_x)
        x = x + self.dropout(_x)  # Residual connection
        
        return x, attention
```

## Putting It All Together

### Complete Encoder

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, ff_dim, 
                 max_seq_length, dropout=0.1):
        super().__init__()
        
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_length, dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(embedding_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = LayerNorm(embedding_dim)
        
    def forward(self, src, src_mask=None):
        # Get token embeddings and add positional encoding
        src = self.token_embedding(src)
        src = self.pos_encoding(src)
        
        # Pass through each encoder layer
        attention_weights = []
        for layer in self.layers:
            src, attention = layer(src, src_mask)
            attention_weights.append(attention)
            
        src = self.norm(src)
        return src, attention_weights
```

### Complete Decoder

```python
class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        # Components
        self.self_attn = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.self_attn_norm = LayerNorm(embedding_dim)
        
        self.enc_attn = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.enc_attn_norm = LayerNorm(embedding_dim)
        
        self.ff = FeedForward(embedding_dim, ff_dim, dropout)
        self.ff_norm = LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, tgt_mask=None, src_tgt_mask=None):
        # Self-attention block
        _x = self.self_attn_norm(x)
        _x, self_attention = self.self_attn(_x, _x, _x, tgt_mask)
        x = x + self.dropout(_x)
        
        # Cross-attention block (attending to encoder outputs)
        _x = self.enc_attn_norm(x)
        _x, enc_attention = self.enc_attn(_x, enc_output, enc_output, src_tgt_mask)
        x = x + self.dropout(_x)
        
        # Feed-forward block
        _x = self.ff_norm(x)
        _x = self.ff(_x)
        x = x + self.dropout(_x)
        
        return x, self_attention, enc_attention

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, ff_dim, 
                 max_seq_length, dropout=0.1):
        super().__init__()
        
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_length, dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(embedding_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = LayerNorm(embedding_dim)
        self.output_linear = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, tgt, enc_output, tgt_mask=None, src_tgt_mask=None):
        # Get token embeddings and add positional encoding
        tgt = self.token_embedding(tgt)
        tgt = self.pos_encoding(tgt)
        
        # Pass through each decoder layer
        self_attentions = []
        enc_attentions = []
        
        for layer in self.layers:
            tgt, self_attn, enc_attn = layer(tgt, enc_output, tgt_mask, src_tgt_mask)
            self_attentions.append(self_attn)
            enc_attentions.append(enc_attn)
            
        tgt = self.norm(tgt)
        output = self.output_linear(tgt)
        
        return output, self_attentions, enc_attentions
```

### Complete Transformer

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim=512, 
                 num_layers=6, num_heads=8, ff_dim=2048, 
                 max_seq_length=512, dropout=0.1):
        super().__init__()
        
        self.encoder = Encoder(
            src_vocab_size, embedding_dim, num_layers, num_heads,
            ff_dim, max_seq_length, dropout
        )
        
        self.decoder = Decoder(
            tgt_vocab_size, embedding_dim, num_layers, num_heads,
            ff_dim, max_seq_length, dropout
        )
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_tgt_mask=None):
        # Process source sequence through encoder
        enc_output, enc_attention = self.encoder(src, src_mask)
        
        # Process target sequence using encoder output
        output, self_attention, cross_attention = self.decoder(
            tgt, enc_output, tgt_mask, src_tgt_mask
        )
        
        return output, enc_attention, self_attention, cross_attention
        
    def generate(self, src, max_length, start_symbol, end_symbol=None):
        """Generate sequence from source input"""
        # Encode source sequence
        src_mask = self._create_pad_mask(src, pad_idx=0)
        enc_output, _ = self.encoder(src, src_mask)
        
        # Start with batch of start symbols
        batch_size = src.shape[0]
        tgt = torch.ones(batch_size, 1).fill_(start_symbol).type_as(src).long()
        
        for i in range(max_length - 1):
            # Create masks for current target sequence
            tgt_mask = self._create_causal_mask(tgt.size(1))
            src_tgt_mask = self._create_pad_mask(src, tgt)
            
            # Make prediction for next token
            output, _, _ = self.decoder(tgt, enc_output, tgt_mask, src_tgt_mask)
            pred = output[:, -1]  # Get predictions for last position
            
            # Get most likely next token
            next_word = pred.argmax(dim=-1).unsqueeze(1)
            
            # Add to sequence
            tgt = torch.cat([tgt, next_word], dim=1)
            
            # Check for end of sequence
            if end_symbol is not None and (next_word == end_symbol).all():
                break
                
        return tgt
    
    def _create_pad_mask(self, seq, pad_idx=0):
        """Create mask for padding tokens"""
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    
    def _create_causal_mask(self, size):
        """Create mask to prevent attention to future tokens"""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return ~mask
```

## Modern Variants and Improvements

Since the original transformer paper, numerous innovations have improved the architecture:

### 1. LayerNorm Position

Pre-norm vs. post-norm placement affects training stability:

```python
# Original (post-norm)
def forward(self, x, mask=None):
    _x, attention = self.self_attn(x, x, x, mask)
    x = x + self.dropout(_x)  # Residual connection
    x = self.norm(x)  # Post-normalization
    return x

# Modern (pre-norm)
def forward(self, x, mask=None):
    _x = self.norm(x)  # Pre-normalization
    _x, attention = self.self_attn(_x, _x, _x, mask)
    x = x + self.dropout(_x)  # Residual connection
    return x
```

### 2. RoPE (Rotary Position Embeddings)

Used in models like GPT-4 and LLaMA, provides better relative positional information:

```python
def get_rotary_embedding(dim, seq_len, device=None):
    """Generate rotary position embeddings"""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    if device is not None:
        inv_freq = inv_freq.to(device)
    
    seq_idx = torch.arange(seq_len).float()
    if device is not None:
        seq_idx = seq_idx.to(device)
    
    sinusoid_inp = torch.einsum("i,j->ij", seq_idx, inv_freq)
    return torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)

def apply_rotary_embedding(query, key, pos_emb):
    """Apply rotary embeddings to queries and keys"""
    # Reshape for rotary computation
    q_cos, q_sin = query[..., 0::2], query[..., 1::2]
    k_cos, k_sin = key[..., 0::2], key[..., 1::2]
    
    # Get position embeddings
    cos_pos, sin_pos = pos_emb[..., 0::2], pos_emb[..., 1::2]
    
    # Apply rotary transformation
    q_rotated = torch.cat([
        q_cos * cos_pos - q_sin * sin_pos,
        q_sin * cos_pos + q_cos * sin_pos
    ], dim=-1)
    
    k_rotated = torch.cat([
        k_cos * cos_pos - k_sin * sin_pos,
        k_sin * cos_pos + k_cos * sin_pos
    ], dim=-1)
    
    return q_rotated, k_rotated
```

### 3. Grouped Query Attention (GQA)

Balances compute efficiency with model quality, used in models like PaLM-2:

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_key_value_heads, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        assert num_heads % num_key_value_heads == 0, "num_heads must be divisible by num_key_value_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_dim // num_heads
        self.num_queries_per_kv = num_heads // num_key_value_heads
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, self.head_dim * num_key_value_heads)
        self.v_proj = nn.Linear(hidden_dim, self.head_dim * num_key_value_heads)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    def forward(self, query, key=None, value=None, mask=None):
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        # Use query as key and value if not provided
        if key is None:
            key = query
        if value is None:
            value = query
            
        # Project and reshape
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, key.size(1), self.num_key_value_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, value.size(1), self.num_key_value_heads, self.head_dim)
        
        # Expand k and v for grouped query attention
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=2)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=2)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(2, 3)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)
            
        # Apply softmax and dropout
        attn_weights = self.dropout(torch.softmax(scores, dim=-1))
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(context)
        
        return output, attn_weights
```

### 4. Multi-Query Attention (MQA)

One key-value projection but multiple query projections:

```python
class MultiQueryAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Multiple query projections, single KV projection
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)  # num_heads * head_dim
        self.k_proj = nn.Linear(hidden_dim, self.head_dim)  # 1 * head_dim
        self.v_proj = nn.Linear(hidden_dim, self.head_dim)  # 1 * head_dim
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    def forward(self, query, key=None, value=None, mask=None):
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        # Use query as key and value if not provided
        if key is None:
            key = query
        if value is None:
            value = query
            
        # Project queries into multiple heads
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Project keys and values into single representation
        k = self.k_proj(key).unsqueeze(2)  # [batch_size, key_len, 1, head_dim]
        v = self.v_proj(value).unsqueeze(2)  # [batch_size, value_len, 1, head_dim]
        
        # Repeat keys and values for each head
        k = k.expand(-1, -1, self.num_heads, -1)  # [batch_size, key_len, num_heads, head_dim]
        v = v.expand(-1, -1, self.num_heads, -1)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(2, 3)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)
            
        # Apply softmax and dropout
        attn_weights = self.dropout(torch.softmax(scores, dim=-1))
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(context)
        
        return output, attn_weights
```

### 5. Flash Attention

A major optimization that reduces memory usage:

```python
# Note: This is a conceptual implementation
# Flash Attention is typically implemented in CUDA for efficiency
def flash_attention(q, k, v, mask=None, dropout_p=0.0):
    """Flash attention implementation conceptually"""
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Scale query
    q = q / math.sqrt(head_dim)
    
    # Initialize output and attention weights
    o = torch.zeros_like(q)
    
    # Process in blocks for memory efficiency
    block_size = 1024  # Example block size
    
    for i in range(0, seq_len, block_size):
        i_end = min(i + block_size, seq_len)
        
        # Load query block
        q_block = q[:, :, i:i_end, :]
        
        for j in range(0, seq_len, block_size):
            j_end = min(j + block_size, seq_len)
            
            # Load key and value blocks
            k_block = k[:, :, j:j_end, :]
            v_block = v[:, :, j:j_end, :]
            
            # Compute attention scores for this block
            scores = torch.matmul(q_block, k_block.transpose(2, 3))
            
            # Apply mask if provided
            if mask is not None:
                block_mask = mask[:, :, i:i_end, j:j_end]
                scores = scores.masked_fill(block_mask == 0, -1e10)
                
            # Apply softmax for this block (simplified - actual implementation computes softmax differently)
            attn_weights = torch.softmax(scores, dim=-1)
            
            if dropout_p > 0 and dropout_p <= 1:
                attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p)
            
            # Update output
            o[:, :, i:i_end, :] += torch.matmul(attn_weights, v_block)
    
    return o
```

## Training and Scaling Transformers

### Optimization Techniques

```python
# Typical training loop with learning rate warmup
def train_transformer(model, train_dataloader, optimizer, criterion, warmup_steps=4000, max_steps=100000):
    """Train a transformer model with learning rate warmup"""
    model.train()
    
    for step, (src, tgt) in enumerate(train_dataloader):
        # Learning rate warmup
        if step < warmup_steps:
            lr = learning_rate * min(1.0, step / warmup_steps)
        else:
            # Learning rate decay
            lr = learning_rate * (max_steps - step) / (max_steps - warmup_steps)
            
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # Forward pass
        src_mask = model._create_pad_mask(src, pad_idx=0)
        tgt_input = tgt[:, :-1]  # Shift right to get teacher forcing inputs
        tgt_output = tgt[:, 1:]  # Shift left to get expected outputs
        
        tgt_mask = model._create_causal_mask(tgt_input.size(1))
        src_tgt_mask = model._create_pad_mask(src, tgt_input)
        
        output, _, _, _ = model(src, tgt_input, src_mask, tgt_mask, src_tgt_mask)
        
        # Calculate loss
        output_flat = output.contiguous().view(-1, output.size(-1))
        tgt_flat = tgt_output.contiguous().view(-1)
        loss = criterion(output_flat, tgt_flat)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Log progress
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}, LR = {lr:.7f}")
```

### Parameter Efficient Fine-tuning (PEFT)

Techniques to adapt large models with minimal computational overhead:

```python
class LoRAModule(nn.Module):
    """Low-Rank Adaptation for efficient fine-tuning"""
    def __init__(self, base_module, rank, alpha=1.0):
        super().__init__()
        self.base_module = base_module
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Input and output dimensions
        self.in_features = base_module.weight.shape[1]
        self.out_features = base_module.weight.shape[0]
        
        # Create low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros((rank, self.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, rank)))
        
        # Initialize low-rank matrices
        nn.init.normal_(self.lora_A, mean=0, std=1)
        nn.init.zeros_(self.lora_B)
        
        # Freeze base module weights
        self.base_module.weight.requires_grad = False
        if hasattr(self.base_module, 'bias') and self.base_module.bias is not None:
            self.base_module.bias.requires_grad = False
            
    def forward(self, x):
        # Original module output
        base_output = self.base_module(x)
        
        # LoRA path
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return base_output + lora_output

# Example usage: applying LoRA to transformer attention layers
def apply_lora_to_transformer(transformer, rank=8, alpha=16):
    """Apply LoRA to the query and value projections of each attention layer"""
    for name, module in transformer.named_modules():
        if isinstance(module, MultiHeadAttention):
            # Apply LoRA to query projection
            module.q_linear = LoRAModule(module.q_linear, rank=rank, alpha=alpha)
            # Apply LoRA to value projection
            module.v_linear = LoRAModule(module.v_linear, rank=rank, alpha=alpha)
    
    return transformer
```

## Practical Usage with Hugging Face Transformers

Modern NLP typically uses pre-implemented libraries like Hugging Face Transformers:

```python
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# Load a pre-trained encoder model
encoder_model_name = "bert-base-uncased"  # Example: BERT
tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
model = AutoModel.from_pretrained(encoder_model_name)

# Encode text
text = "The transformer architecture revolutionized natural language processing."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# Get embeddings
embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

# Load a pre-trained decoder-only language model
lm_model_name = "gpt2"  # Example: GPT-2
lm_tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
lm_model = AutoModelForCausalLM.from_pretrained(lm_model_name)

# Generate text
prompt = "The transformer architecture is powerful because"
inputs = lm_tokenizer(prompt, return_tensors="pt")
outputs = lm_model.generate(
    inputs.input_ids,
    max_length=50,
    num_return_sequences=1,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

# Decode the generated text
generated_text = lm_tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## The Impact of Transformers

Transformers have enabled countless breakthroughs in AI:

1. **Large Language Models**: GPT, LLaMA, Claude
2. **Vision Transformers**: ViT, CLIP
3. **Multimodal Models**: GPT-4V, Gemini, Claude Opus
4. **Diffusion Models**: Stable Diffusion (using transformers for conditional generation)

The architecture continues to evolve, with ongoing research into:

- **Efficient Transformers**: Reducing computational costs
- **Sparse Attention**: Handling longer contexts
- **Improved Training**: Better optimization and stability
- **Memory Mechanisms**: Enhancing long-term reasoning abilities

## Conclusion

The transformer architecture has fundamentally changed AI by providing a foundation for models that can understand and generate human-like text, images, and more. By understanding the key components and mechanisms of transformers, you gain insight into the core technology powering the current wave of generative AI systems.

As research continues, transformers will likely evolve further, but their central innovations—parallel processing, self-attention, and flexible architecture—will continue to influence AI system design for years to come.