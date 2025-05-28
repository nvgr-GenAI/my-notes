---
title: Basics of Language Modeling
sidebar_position: 2
description: Fundamentals of language modeling, tokens, tokenization, and LLM input/output patterns
---

# Basics of Language Modeling

Before diving into the complexities of large language models, it's essential to understand the fundamental concept that underlies them all: language modeling. This page covers the core concepts that form the foundation of how modern LLMs work.

## What is Language Modeling?

Language modeling is the task of assigning probabilities to sequences of words or tokens. More formally, a language model computes the probability distribution over a sequence of tokens.

### Mathematical Definition

In mathematical terms, given a sequence of tokens $(t_1, t_2, ..., t_{n-1})$, a language model computes the probability of the next token $t_n$:

$$P(t_n | t_1, t_2, ..., t_{n-1})$$

For a complete sequence, the joint probability is calculated as:

$$P(t_1, t_2, ..., t_n) = \prod_{i=1}^{n} P(t_i | t_1, t_2, ..., t_{i-1})$$

This is also known as the **chain rule of probability** for sequences.

### Historical Evolution

Language modeling has evolved significantly over time:

1. **Statistical n-gram models** (1980s-2000s)
   - Count-based approaches using Markov assumptions
   - Limited to short contexts (typically 3-5 tokens)
   ```python
   # Simplified trigram model
   def trigram_probability(word3, word1, word2, corpus):
       bigram_count = count_sequence([word1, word2], corpus)
       trigram_count = count_sequence([word1, word2, word3], corpus)
       return trigram_count / bigram_count if bigram_count > 0 else 0
   ```

2. **Feed-forward neural language models** (2000s)
   - Neural networks with fixed context windows
   - Word embeddings to capture semantic relationships
   ```python
   # Simple feed-forward neural LM
   class FFLanguageModel(nn.Module):
       def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
           super().__init__()
           self.embeddings = nn.Embedding(vocab_size, embedding_dim)
           self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
           self.linear2 = nn.Linear(hidden_dim, vocab_size)
           
       def forward(self, inputs):
           embeds = self.embeddings(inputs).view((inputs.shape[0], -1))
           hidden = torch.tanh(self.linear1(embeds))
           output = self.linear2(hidden)
           return F.log_softmax(output, dim=1)
   ```

3. **Recurrent neural networks** (2010s)
   - LSTM and GRU models handling variable-length sequences
   - Improved handling of long-range dependencies
   ```python
   # LSTM-based language model
   class LSTMLanguageModel(nn.Module):
       def __init__(self, vocab_size, embedding_dim, hidden_dim):
           super().__init__()
           self.embeddings = nn.Embedding(vocab_size, embedding_dim)
           self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
           self.linear = nn.Linear(hidden_dim, vocab_size)
           
       def forward(self, inputs):
           embeds = self.embeddings(inputs)
           lstm_out, _ = self.lstm(embeds)
           output = self.linear(lstm_out)
           return output
   ```

4. **Transformers and self-attention** (2017-present)
   - Parallel processing with attention mechanisms
   - State-of-the-art performance with scaling
   ```python
   # Simplified transformer decoder block
   class TransformerDecoderBlock(nn.Module):
       def __init__(self, d_model, num_heads, d_ff):
           super().__init__()
           self.self_attn = nn.MultiheadAttention(d_model, num_heads)
           self.feed_forward = nn.Sequential(
               nn.Linear(d_model, d_ff),
               nn.ReLU(),
               nn.Linear(d_ff, d_model)
           )
           self.norm1 = nn.LayerNorm(d_model)
           self.norm2 = nn.LayerNorm(d_model)
           
       def forward(self, x, mask):
           # Self attention with residual connection and normalization
           attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
           x = self.norm1(x + attn_output)
           
           # Feed forward with residual connection and normalization
           ff_output = self.feed_forward(x)
           x = self.norm2(x + ff_output)
           return x
   ```

### Types of Language Models

Language models can be categorized based on their directionality and architecture:

1. **Autoregressive (left-to-right) models**
   - Generate one token at a time from left to right
   - Examples: GPT family, LLaMA, Claude
   - Used for text generation tasks

2. **Masked language models**
   - Predict masked tokens within a bidirectional context
   - Examples: BERT, RoBERTa
   - Used primarily for understanding and classification tasks

3. **Encoder-decoder models**
   - Encode input sequence, then decode to output sequence
   - Examples: T5, BART
   - Used for translation, summarization, and other text-to-text tasks

## Overview of Tokens and Tokenization

Language models don't process raw text directly but work with numerical representations called tokens. Tokenization is the process of converting text into these tokens.

### What Are Tokens?

Tokens are the fundamental units that language models process:
- Can represent characters, subwords, or whole words
- Are converted to numerical IDs for model processing
- Define the vocabulary size of the model

### Common Tokenization Methods

1. **Word-level tokenization**
   - Splits text by word boundaries (spaces, punctuation)
   - Creates large vocabularies with many rare words
   ```python
   def simple_word_tokenizer(text):
       # Split by whitespace and punctuation
       return re.findall(r'\b\w+\b|[^\w\s]', text)
   ```

2. **Character-level tokenization**
   - Uses individual characters as tokens
   - Small vocabulary but long sequences
   ```python
   def char_tokenizer(text):
       return list(text)  # Split into individual characters
   ```

3. **Subword tokenization (most common in modern LLMs)**
   - Balances word and character approaches
   - Handles rare words by splitting into common subunits
   - Three popular algorithms:

   **a. Byte-Pair Encoding (BPE)**
   ```python
   # Simplified BPE tokenization algorithm
   def byte_pair_encoding_train(text, vocab_size):
       # Start with character vocabulary
       vocab = set(char for char in text)
       
       # Split text into characters
       words = text.split()
       splits = {word: [c for c in word] for word in words}
       
       # Merge most frequent pairs until reaching vocab_size
       while len(vocab) < vocab_size:
           # Count pairs
           pair_counts = {}
           for word_splits in splits.values():
               for i in range(len(word_splits) - 1):
                   pair = (word_splits[i], word_splits[i + 1])
                   pair_counts[pair] = pair_counts.get(pair, 0) + 1
           
           if not pair_counts:
               break
           
           # Find most frequent pair
           best_pair = max(pair_counts, key=pair_counts.get)
           merged_token = best_pair[0] + best_pair[1]
           vocab.add(merged_token)
           
           # Update splits
           for word, word_splits in splits.items():
               i = 0
               while i < len(word_splits) - 1:
                   if (word_splits[i], word_splits[i + 1]) == best_pair:
                       word_splits[i:i+2] = [merged_token]
                   else:
                       i += 1
       
       return vocab
   ```

   **b. WordPiece (used by BERT)**
   ```python
   # WordPiece differs from BPE by using likelihood rather than frequency
   # This is a simplified conceptual version
   def wordpiece_train(text, vocab_size):
       # Similar to BPE but uses different scoring mechanism
       # for choosing which pieces to merge
       pass  # Full implementation omitted for brevity
   ```

   **c. SentencePiece / Unigram (used by many multilingual models)**
   ```python
   # Unigram model starts with large vocabulary and removes tokens
   # Based on likelihood maximization
   def unigram_train(text, vocab_size):
       # Start with a large vocabulary
       vocab = initialize_large_vocab(text)
       
       while len(vocab) > vocab_size:
           # Calculate loss change for each token if removed
           losses = {}
           for token in vocab:
               losses[token] = calculate_loss_change_if_removed(token, text, vocab)
           
           # Remove token with smallest loss impact
           token_to_remove = min(losses, key=losses.get)
           vocab.remove(token_to_remove)
       
       return vocab
   ```

### Tokenization Example

Let's see how a sentence might be tokenized with BPE:

```
Input: "The transformer architecture revolutionized NLP."

Tokens (GPT-2 tokenizer):
['The', 'Ġtransformer', 'Ġarchitecture', 'Ġrevolution', 'ized', 'ĠN', 'LP', '.']

Token IDs:
[464, 11337, 4950, 6579, 287, 499, 22802, 13]
```

Notice how "revolutionized" is split into "revolution" + "ized", showing how subword tokenization handles uncommon words.

### Special Tokens

Most tokenizers include special tokens for specific purposes:

- **BOS (Beginning of Sequence)**: Marks the start of text (e.g., `<s>`)
- **EOS (End of Sequence)**: Marks the end of text (e.g., `</s>`)
- **PAD**: Used to make sequences uniform length (e.g., `<pad>`)
- **UNK (Unknown)**: Represents tokens not in vocabulary (e.g., `<unk>`)
- **Mask**: For masked language modeling (e.g., `<mask>`)

```python
# Adding special tokens to tokenized text
def add_special_tokens(token_ids):
    BOS_ID = 1  # Example ID
    EOS_ID = 2  # Example ID
    return [BOS_ID] + token_ids + [EOS_ID]
```

### Tokenization Limitations

Tokenization has several important limitations to be aware of:

1. **Language bias**: Most tokenizers are trained primarily on English and related languages
2. **Vocabulary size tradeoffs**: Larger vocabularies use fewer tokens per word but require more parameters
3. **Out-of-vocabulary handling**: Rare words may be split suboptimally
4. **Context window consumption**: More tokens means less effective context window

## Understanding the Input/Output of LLMs

Now that we understand language modeling and tokenization, let's examine how LLMs process inputs and generate outputs.

### Input Processing Flow

The typical processing flow for inputs to an LLM:

1. **Text preprocessing**
   ```python
   def preprocess_text(text):
       # Basic preprocessing
       text = text.strip()
       # Remove excessive whitespace
       text = re.sub(r'\s+', ' ', text)
       return text
   ```

2. **Tokenization**
   ```python
   def tokenize_input(text, tokenizer):
       tokens = tokenizer.encode(text)
       return tokens
   ```

3. **Context formatting**
   ```python
   def format_context(prompt, max_length):
       # Format based on model requirements
       tokens = tokenize_input(prompt)
       
       # Handle context window limits
       if len(tokens) > max_length:
           tokens = tokens[-max_length:]  # Keep most recent tokens
           
       return tokens
   ```

4. **Input embedding**
   ```python
   def embed_input(tokens, embedding_layer):
       # Convert token IDs to embeddings
       return embedding_layer(tokens)
   ```

5. **Positional encoding**
   ```python
   def add_positional_encoding(embeddings, max_len, d_model):
       positions = torch.arange(max_len).unsqueeze(1)
       div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
       
       pos_encoding = torch.zeros_like(embeddings)
       pos_encoding[:, 0::2] = torch.sin(positions * div_term)
       pos_encoding[:, 1::2] = torch.cos(positions * div_term)
       
       return embeddings + pos_encoding
   ```

### Generation Process

The generation process typically follows these steps:

1. **Forward pass through model layers**
   ```python
   def forward_pass(model, input_embeddings, attention_mask):
       # Process through transformer layers
       hidden_states = input_embeddings
       for layer in model.layers:
           hidden_states = layer(hidden_states, attention_mask)
       return hidden_states
   ```

2. **Output projection to vocabulary**
   ```python
   def project_to_vocab(hidden_states, output_layer):
       # Project final hidden states to vocabulary logits
       logits = output_layer(hidden_states)
       return logits
   ```

3. **Sampling for next token**
   ```python
   def sample_next_token(logits, temperature=1.0, top_k=0, top_p=0.9):
       # Apply temperature
       logits = logits / temperature
       
       # Apply top-k filtering
       if top_k > 0:
           top_k_logits, top_k_indices = torch.topk(logits, k=top_k)
           logits = torch.full_like(logits, float('-inf'))
           logits.scatter_(1, top_k_indices, top_k_logits)
       
       # Apply top-p (nucleus) sampling
       if 0 < top_p < 1.0:
           sorted_logits, sorted_indices = torch.sort(logits, descending=True)
           cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
           
           # Remove tokens with cumulative probability above the threshold
           sorted_indices_to_remove = cumulative_probs > top_p
           # Shift the indices to the right to keep also the first token above the threshold
           sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
           sorted_indices_to_remove[..., 0] = 0
           
           # Scatter sorted tensors to original indexing
           indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
           logits[indices_to_remove] = float('-inf')
       
       # Convert to probabilities and sample
       probs = F.softmax(logits, dim=-1)
       next_token = torch.multinomial(probs, num_samples=1)
       
       return next_token
   ```

4. **Autoregressive generation**
   ```python
   def generate_text(model, prompt_tokens, max_new_tokens, **sampling_params):
       # Start with prompt tokens
       generated = prompt_tokens.clone()
       
       # Generate one token at a time
       for _ in range(max_new_tokens):
           # Get predictions for next token (using only the last token or all tokens depending on model)
           outputs = model(generated)
           next_token_logits = outputs[:, -1, :]  # Get logits for the last position
           
           # Sample next token
           next_token = sample_next_token(next_token_logits, **sampling_params)
           
           # Append to generated sequence
           generated = torch.cat([generated, next_token], dim=1)
           
           # Check for end of sequence token
           if next_token.item() == tokenizer.eos_token_id:
               break
               
       return generated
   ```

### Controlling Generation

Modern LLMs provide several parameters to control the generation process:

#### Temperature

Controls randomness in output generation:

```python
def adjust_logits_with_temperature(logits, temperature):
    """
    Higher temperature (>1.0) = more random outputs
    Lower temperature (<1.0) = more deterministic outputs
    Temperature of 0 = greedy sampling (always pick highest probability)
    """
    if temperature == 0:
        # Greedy sampling
        return torch.argmax(logits, dim=-1).unsqueeze(-1)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
```

#### Top-k Sampling

Limits sampling to the k most likely tokens:

```python
def top_k_sampling(logits, k=50):
    """
    Only sample from the top k most likely tokens
    """
    top_k_logits, top_k_indices = torch.topk(logits, k=k)
    
    # Create a mask for the top-k tokens
    mask = torch.zeros_like(logits).scatter_(1, top_k_indices, 1)
    logits = logits * mask + (-1e10) * (1 - mask)
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

#### Nucleus (Top-p) Sampling

Samples from the smallest set of tokens whose cumulative probability exceeds p:

```python
def nucleus_sampling(logits, p=0.9):
    """
    Sample from the smallest set of tokens that exceed probability p
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

#### Beam Search

Unlike sampling, beam search tries to find the most likely sequence overall:

```python
def beam_search(model, initial_tokens, beam_width=5, max_length=50):
    """
    Beam search to find the most likely sequence
    """
    # Start with initial tokens
    sequences = [(initial_tokens, 0)]  # (sequence, score)
    
    for _ in range(max_length):
        candidates = []
        
        # Expand each current sequence
        for seq, score in sequences:
            if seq[-1] == tokenizer.eos_token_id:
                # Keep completed sequences
                candidates.append((seq, score))
                continue
                
            # Get predictions for next token
            outputs = model(seq)
            next_token_logits = outputs[:, -1, :]
            
            # Get top-k next tokens
            next_token_logits = F.log_softmax(next_token_logits, dim=-1)
            top_k_logits, top_k_indices = torch.topk(next_token_logits, k=beam_width)
            
            # Create new candidate sequences
            for logit, idx in zip(top_k_logits[0], top_k_indices[0]):
                new_seq = torch.cat([seq, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                new_score = score + logit.item()
                candidates.append((new_seq, new_score))
        
        # Select top beam_width candidates
        sequences = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Check if all sequences are completed
        if all(seq[-1] == tokenizer.eos_token_id for seq, _ in sequences):
            break
    
    # Return the highest scoring sequence
    return sequences[0][0]
```

### Prompt Engineering

The way you structure prompts significantly impacts LLM output. Common techniques include:

1. **Zero-shot prompting**
   ```
   Classify the following text as positive, negative, or neutral: 
   "The food was delicious but the service was slow."
   ```

2. **Few-shot prompting**
   ```
   Classify the sentiment of the text:
   
   Text: "I loved the movie, it was fantastic!"
   Sentiment: Positive
   
   Text: "The restaurant was too noisy and the food was mediocre."
   Sentiment: Negative
   
   Text: "The package arrived on the expected delivery date."
   Sentiment: Neutral
   
   Text: "The hotel staff was rude and our room wasn't ready."
   Sentiment:
   ```

3. **Chain-of-thought prompting**
   ```
   Question: A store has 10 shirts. They sell 3 on Monday and twice as many on Tuesday. How many shirts are left?
   
   Let me think through this step by step:
   1. The store starts with 10 shirts.
   2. On Monday, they sell 3 shirts. Now they have 10 - 3 = 7 shirts.
   3. On Tuesday, they sell twice as many as Monday, which is 2 × 3 = 6 shirts.
   4. After Tuesday, they have 7 - 6 = 1 shirt left.
   
   The answer is 1 shirt.
   ```

4. **System instructions** (for models that support them)
   ```
   System: You are a helpful assistant that translates English to French.
   User: How do you say "The weather is beautiful today" in French?
   ```

## Input/Output Structure in Different LLM Systems

Different LLM systems structure their inputs and outputs in specific ways:

### OpenAI API Structure

```python
import openai

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7,
    max_tokens=150
)

print(response['choices'][0]['message']['content'])
```

### Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

inputs = tokenizer("The capital of France is", return_tensors="pt")
outputs = model.generate(
    inputs.input_ids, 
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### LangChain Framework

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.7)

prompt = PromptTemplate(
    input_variables=["country"],
    template="What is the capital of {country}?"
)

final_prompt = prompt.format(country="France")
print(llm(final_prompt))
```

## Conclusion

Understanding the basics of language modeling and how tokens are processed by LLMs provides a solid foundation for working with these systems effectively. The next sections will build on these concepts to explore the specific architectures that power modern LLMs and how they are trained and used.

In the following sections, we'll dive deeper into the transformer architecture that revolutionized language modeling and explore how these models are trained at scale.