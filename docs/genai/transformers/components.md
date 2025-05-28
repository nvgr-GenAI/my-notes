---
title: Transformer Components
sidebar_position: 2
description: Detailed explanation of core transformer architecture components
---

# Transformer Components

Transformers have become the foundation of modern natural language processing and generative AI. First introduced in the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. in 2017, transformers revolutionized the field by eliminating the need for recurrence or convolution in sequence models while enabling parallelized training.

## Core Components of Transformer Architecture

The transformer architecture consists of several key components that work together to process and generate sequences effectively.

### 1. Input and Output Embeddings

Embeddings convert tokens (words, subwords, or characters) into dense vectors that capture semantic meaning:

```python
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        """
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the embedding vectors
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
    
    def forward(self, tokens):
        """
        Args:
            tokens: Tensor of token indices [batch_size, seq_len]
            
        Returns:
            Token embeddings [batch_size, seq_len, embedding_dim]
        """
        # Scale embeddings by sqrt(d_model) as in the original paper
        return self.embedding(tokens) * torch.sqrt(torch.tensor(self.embedding_dim, dtype=torch.float32))
```

### 2. Positional Encoding

Since transformers process sequences in parallel rather than sequentially, they need positional information to understand token order:

```python
import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        """
        Args:
            d_model: Dimension of the model
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix [max_seq_length, d_model]
        pe = torch.zeros(max_seq_length, d_model)
        
        # Create a tensor representing positions
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Create a tensor representing the different dimensions
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        
        # Apply sine to even positions and cosine to odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension [1, max_seq_length, d_model]
        pe = pe.unsqueeze(0)
        
        # Register buffer (persistent state) that won't be considered a model parameter
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Embeddings [batch_size, seq_len, embedding_dim]
            
        Returns:
            Embeddings + positional encodings
        """
        # Add positional encoding to embeddings (only up to the sequence length)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

Visualizing positional encoding patterns:

```python
import matplotlib.pyplot as plt

def visualize_positional_encoding():
    """Visualize positional encodings for the first few positions and dimensions"""
    model_dim = 128
    pos_encoding = PositionalEncoding(model_dim, 100)
    
    # Create a dummy tensor just to get the positional encoding
    dummy_input = torch.zeros(1, 100, model_dim)
    pos_enc = pos_encoding(dummy_input)[0]  # [seq_len, d_model]
    
    # Plot the first 20 positions for the first 20 dimensions
    plt.figure(figsize=(12, 8))
    plt.imshow(pos_enc[:20, :20].detach().numpy(), cmap='viridis', aspect='auto')
    plt.xlabel('Dimension')
    plt.ylabel('Position')
    plt.colorbar()
    plt.title('Positional Encoding Visualization')
    plt.savefig('positional_encoding.png')
    plt.close()
```

### 3. Self-Attention Mechanism

The core innovation of transformers is the self-attention mechanism, which allows the model to focus on different parts of the input sequence:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V and output
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x):
        """
        Split the last dimension into (heads, depth)
        Args:
            x: tensor with shape [batch_size, seq_len, d_model]
            
        Returns:
            tensor with shape [batch_size, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
    
    def merge_heads(self, x):
        """
        Merge the heads back together
        Args:
            x: tensor with shape [batch_size, num_heads, seq_len, d_k]
            
        Returns:
            tensor with shape [batch_size, seq_len, d_model]
        """
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2)  # [batch_size, seq_len, num_heads, d_k]
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute the scaled dot product attention
        
        Args:
            Q: Query tensor [batch_size, num_heads, seq_len_q, d_k]
            K: Key tensor [batch_size, num_heads, seq_len_k, d_k]
            V: Value tensor [batch_size, num_heads, seq_len_v, d_k]
            mask: Optional mask [batch_size, 1, seq_len_q, seq_len_k]
            
        Returns:
            context vector and attention weights
        """
        # Compute Q·K^T
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch_size, num_heads, seq_len_q, seq_len_k]
        
        # Scale by sqrt(d_k)
        scores = scores / math.sqrt(self.d_k)
        
        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Multiply by V to get context vectors
        context = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len_q, d_k]
        
        return context, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: Query tensor [batch_size, seq_len_q, d_model]
            key: Key tensor [batch_size, seq_len_k, d_model]
            value: Value tensor [batch_size, seq_len_v, d_model]
            mask: Optional mask [batch_size, 1, seq_len_q, seq_len_k]
            
        Returns:
            Multi-head attention output [batch_size, seq_len_q, d_model]
        """
        batch_size = query.size(0)
        
        # Linear projections and split heads
        Q = self.split_heads(self.w_q(query))  # [batch_size, num_heads, seq_len_q, d_k]
        K = self.split_heads(self.w_k(key))    # [batch_size, num_heads, seq_len_k, d_k]
        V = self.split_heads(self.w_v(value))  # [batch_size, num_heads, seq_len_v, d_k]
        
        # Apply scaled dot product attention
        context, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Merge heads and apply final linear layer
        output = self.merge_heads(context)  # [batch_size, seq_len_q, d_model]
        output = self.w_o(output)
        
        return output, attention_weights
```

#### Visualizing Self-Attention

To understand how self-attention works, it's helpful to visualize the attention weights:

```python
def visualize_attention(attention_weights, tokens):
    """
    Visualize attention weights between tokens
    
    Args:
        attention_weights: tensor of shape [batch_size, num_heads, seq_len_q, seq_len_k]
        tokens: list of input tokens
    """
    import seaborn as sns
    
    # Take the first example in batch and first attention head
    att_mat = attention_weights[0, 0].detach().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(att_mat, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.title('Self-Attention Weights')
    plt.savefig('self_attention_visualization.png')
    plt.close()
```

### 4. Feed-Forward Networks

Each attention block is followed by a position-wise feed-forward network:

```python
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: Model dimension (input and output)
            d_ff: Hidden dimension of the feed-forward network
            dropout: Dropout probability
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # Modern transformers typically use GELU instead of ReLU
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Processed tensor [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
```

### 5. Layer Normalization

Layer normalization is applied after each sub-layer in the transformer:

```python
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        """
        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))  # Scale parameter
        self.beta = nn.Parameter(torch.zeros(d_model))  # Shift parameter
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Normalized tensor [batch_size, seq_len, d_model]
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

### 6. Residual Connections

Residual connections help with the flow of gradients through the deep network:

```python
class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        """
        Apply residual connection with sublayer
        
        Args:
            x: Input tensor
            sublayer: Function/layer to apply
            
        Returns:
            Output with residual connection and normalization
        """
        # Apply Pre-LN architecture (LayerNorm before sublayer)
        # Modern transformers often use Pre-LN instead of Post-LN for better stability
        return x + self.dropout(sublayer(self.norm(x)))
```

## Encoder and Decoder Architectures

Transformers can have encoder-only, decoder-only, or encoder-decoder architectures.

### 1. Encoder Block

An encoder block consists of self-attention followed by a feed-forward network, both with residual connections:

```python
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Hidden dimension of feed-forward network
            dropout: Dropout probability
        """
        super().__init__()
        
        # Self-attention and feed-forward layers
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # Residual connections with layer norm
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional mask for self-attention
            
        Returns:
            Processed tensor [batch_size, seq_len, d_model]
        """
        # Apply self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.residual1(x, lambda x: attn_output)
        
        # Apply feed-forward with residual connection
        x = self.residual2(x, self.feed_forward)
        
        return x
```

### 2. Decoder Block

A decoder block consists of masked self-attention, cross-attention to the encoder output, and a feed-forward network:

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Hidden dimension of feed-forward network
            dropout: Dropout probability
        """
        super().__init__()
        
        # Self-attention, cross-attention, and feed-forward layers
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # Residual connections with layer norm
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        self.residual3 = ResidualConnection(d_model, dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Target tensor [batch_size, tgt_seq_len, d_model]
            encoder_output: Encoder output [batch_size, src_seq_len, d_model]
            src_mask: Mask for encoder attention
            tgt_mask: Mask for decoder self-attention (usually to prevent looking ahead)
            
        Returns:
            Processed tensor [batch_size, tgt_seq_len, d_model]
        """
        # Self-attention with residual connection (masked to prevent looking ahead)
        self_attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.residual1(x, lambda x: self_attn_output)
        
        # Cross-attention with encoder output
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.residual2(x, lambda x: cross_attn_output)
        
        # Feed-forward network with residual connection
        x = self.residual3(x, self.feed_forward)
        
        return x
```

### 3. Complete Encoder

The full encoder consists of multiple encoder blocks stacked together:

```python
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1, max_seq_len=5000):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Hidden dimension of feed-forward network
            num_layers: Number of encoder blocks
            dropout: Dropout probability
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()
        
        # Embedding and positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Stack of encoder blocks
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = LayerNormalization(d_model)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input embeddings [batch_size, src_seq_len, d_model]
            mask: Mask for self-attention
            
        Returns:
            Encoder output [batch_size, src_seq_len, d_model]
        """
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through each encoder block
        for layer in self.layers:
            x = layer(x, mask)
        
        # Apply final layer normalization
        return self.norm(x)
```

### 4. Complete Decoder

The full decoder consists of multiple decoder blocks stacked together:

```python
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1, max_seq_len=5000):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Hidden dimension of feed-forward network
            num_layers: Number of decoder blocks
            dropout: Dropout probability
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()
        
        # Embedding and positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Stack of decoder blocks
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = LayerNormalization(d_model)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Target embeddings [batch_size, tgt_seq_len, d_model]
            encoder_output: Output from the encoder [batch_size, src_seq_len, d_model]
            src_mask: Mask for encoder attention
            tgt_mask: Mask for decoder self-attention (usually to prevent looking ahead)
            
        Returns:
            Decoder output [batch_size, tgt_seq_len, d_model]
        """
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through each decoder block
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        # Apply final layer normalization
        return self.norm(x)
```

## Full Transformer Architecture

Putting everything together for a full encoder-decoder transformer:

```python
class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 tgt_vocab_size, 
                 d_model=512, 
                 num_heads=8, 
                 d_ff=2048, 
                 num_layers=6, 
                 dropout=0.1,
                 max_seq_len=5000):
        """
        Args:
            src_vocab_size: Size of source vocabulary
            tgt_vocab_size: Size of target vocabulary
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Hidden dimension of feed-forward network
            num_layers: Number of encoder/decoder blocks
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        # Token embeddings
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)
        
        # Encoder and decoder
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout, max_seq_len)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout, max_seq_len)
        
        # Final linear layer and softmax for prediction
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        
    def encode(self, src, src_mask=None):
        """
        Encode the source sequence
        
        Args:
            src: Source token indices [batch_size, src_seq_len]
            src_mask: Mask for source sequence
            
        Returns:
            Encoder output
        """
        src_embedded = self.src_embedding(src)
        return self.encoder(src_embedded, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        Decode with target sequence and encoder output
        
        Args:
            tgt: Target token indices [batch_size, tgt_seq_len]
            encoder_output: Output from the encoder
            src_mask: Mask for source sequence
            tgt_mask: Mask for target sequence (usually to prevent looking ahead)
            
        Returns:
            Decoder output
        """
        tgt_embedded = self.tgt_embedding(tgt)
        return self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass through the transformer
        
        Args:
            src: Source token indices [batch_size, src_seq_len]
            tgt: Target token indices [batch_size, tgt_seq_len]
            src_mask: Mask for source sequence
            tgt_mask: Mask for target sequence (usually to prevent looking ahead)
            
        Returns:
            Output logits [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        # Generate encoder output
        encoder_output = self.encode(src, src_mask)
        
        # Generate decoder output
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary size
        output = self.linear(decoder_output)
        
        return output
```

## Common Architecture Variants

### 1. Encoder-Only Models (BERT, RoBERTa, etc.)

Encoder-only models use only the encoder part of the transformer. They are bidirectional and typically used for tasks like classification and named entity recognition.

```python
class BERTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, d_ff=3072, num_layers=12, dropout=0.1, max_seq_len=512):
        super().__init__()
        
        # Token embedding
        self.embedding = TokenEmbedding(vocab_size, d_model)
        
        # Encoder
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout, max_seq_len)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: Input token indices [batch_size, seq_len]
            attention_mask: Attention mask for padding [batch_size, seq_len]
            
        Returns:
            Encoder output [batch_size, seq_len, d_model]
        """
        # Convert attention mask to proper format
        if attention_mask is not None:
            # Convert from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Embed tokens
        embedded = self.embedding(input_ids)
        
        # Pass through encoder
        return self.encoder(embedded, attention_mask)
```

### 2. Decoder-Only Models (GPT, LLaMA, etc.)

Decoder-only models use only the decoder part, but without cross-attention since there's no encoder. They use causal (masked) self-attention to prevent looking at future tokens.

```python
class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, d_ff=3072, num_layers=12, dropout=0.1, max_seq_len=1024):
        super().__init__()
        
        # Token embedding
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # GPT uses decoder blocks but without cross-attention
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = LayerNormalization(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
    def generate_causal_mask(self, seq_len):
        """
        Generate a causal mask for autoregressive decoding
        
        Args:
            seq_len: Length of sequence
            
        Returns:
            Causal mask [1, 1, seq_len, seq_len]
        """
        # Create a mask that prevents attending to future tokens
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        return ~mask  # Invert so that 1 = attend, 0 = don't attend
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: Input token indices [batch_size, seq_len]
            attention_mask: Attention mask for padding [batch_size, seq_len]
            
        Returns:
            Output logits [batch_size, seq_len, vocab_size]
        """
        seq_len = input_ids.size(1)
        
        # Create causal mask
        causal_mask = self.generate_causal_mask(seq_len).to(input_ids.device)
        
        # If there's an additional padding mask, combine it with the causal mask
        if attention_mask is not None:
            # Convert from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            causal_mask = causal_mask & attention_mask
        
        # Embed tokens with positional encoding
        x = self.positional_encoding(self.embedding(input_ids))
        
        # Pass through each transformer block
        for block in self.blocks:
            x = block(x, causal_mask)
        
        # Apply final layer norm and projection to vocabulary
        x = self.norm(x)
        logits = self.head(x)
        
        return logits
```

### 3. Encoder-Decoder Models (T5, BART, etc.)

Encoder-decoder models use both components, similar to the original transformer architecture.

```python
class T5Model(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 d_model=512, 
                 num_heads=8, 
                 d_ff=2048, 
                 num_encoder_layers=6,
                 num_decoder_layers=6, 
                 dropout=0.1,
                 max_seq_len=512):
        super().__init__()
        
        # Shared embedding between encoder and decoder
        self.shared_embedding = TokenEmbedding(vocab_size, d_model)
        
        # Encoder and decoder
        self.encoder = Encoder(d_model, num_heads, d_ff, num_encoder_layers, dropout, max_seq_len)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_decoder_layers, dropout, max_seq_len)
        
        # Final linear layer
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Share weights between embedding and lm_head
        self.lm_head.weight = self.shared_embedding.embedding.weight
        
    def forward(self, 
                input_ids, 
                decoder_input_ids,
                encoder_attention_mask=None,
                decoder_attention_mask=None):
        """
        Args:
            input_ids: Encoder input token indices [batch_size, src_seq_len]
            decoder_input_ids: Decoder input token indices [batch_size, tgt_seq_len]
            encoder_attention_mask: Mask for encoder inputs [batch_size, src_seq_len]
            decoder_attention_mask: Mask for decoder inputs [batch_size, tgt_seq_len]
            
        Returns:
            Output logits [batch_size, tgt_seq_len, vocab_size]
        """
        # Convert attention masks to proper format
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Create causal mask for decoder
        seq_len = decoder_input_ids.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(decoder_input_ids.device)
        causal_mask = ~causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_seq_len, tgt_seq_len]
        
        # If there's an additional padding mask, combine it with the causal mask
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.unsqueeze(1).unsqueeze(2)
            causal_mask = causal_mask & decoder_attention_mask
        
        # Encode
        encoder_output = self.encoder(
            self.shared_embedding(input_ids), 
            encoder_attention_mask
        )
        
        # Decode
        decoder_output = self.decoder(
            self.shared_embedding(decoder_input_ids),
            encoder_output,
            encoder_attention_mask,
            causal_mask
        )
        
        # Project to vocabulary
        logits = self.lm_head(decoder_output)
        
        return logits
```

## Common Workloads and Model Choices

Different transformer architectures excel at different types of tasks:

| Task Type | Recommended Architecture | Example Models |
|-----------|--------------------------|----------------|
| Text Classification | Encoder-only | BERT, RoBERTa, DeBERTa |
| Named Entity Recognition | Encoder-only | BERT, SpanBERT |
| Text Generation | Decoder-only | GPT-3/4, LLaMA, Claude |
| Summarization | Encoder-decoder or Decoder-only | BART, T5, GPT-3/4 |
| Translation | Encoder-decoder | T5, mT5, NLLB |
| Question Answering | Encoder-only or Encoder-decoder | BERT, T5, Flan-T5 |
| Code Generation | Decoder-only | Codex, CodeLLaMA |

## Implementation Tips

1. **Memory Optimization**
   - Use mixed precision (FP16/BF16) to reduce memory footprint
   - Implement gradient checkpointing for backprop-intensive workloads
   - Consider Flash Attention for more efficient attention computation

2. **Computational Efficiency**
   - Use Multi-Query or Grouped-Query attention to reduce KV cache size
   - Implement model parallelism for very large models
   - Consider smaller, specialized models for specific tasks

3. **Training Stability**
   - Pre-LN architecture can be more stable than Post-LN during training
   - Gradually increase model and batch size during training
   - Use learning rate warmup followed by cosine decay

4. **Performance Evaluation**
   - Evaluate perplexity for language modeling tasks
   - Use task-specific metrics for downstream applications (BLEU, ROUGE, exact match, etc.)
   - Consider human evaluation for generation quality

## Conclusion

The transformer architecture has revolutionized natural language processing and artificial intelligence more broadly. Its adaptability has enabled different variants to excel at different tasks, from bidirectional understanding to autoregressive generation. By understanding the core components and architectural variations, you can better select, implement, and optimize transformer models for your specific use cases.

As research continues to evolve, we're seeing new innovations like more efficient attention mechanisms, specialized architectures, and hybrid approaches that combine the strengths of different transformer variants. Staying current with these developments can help you build more effective and efficient AI systems.