"""
Advanced Neural Network Architecture for Contextual Text Processing
Transformer-based model with attention mechanisms optimized for production
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Dropout, Linear, Embedding
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism with optimizations"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = Linear(d_model, d_model, bias=False)
        self.w_k = Linear(d_model, d_model, bias=False)
        self.w_v = Linear(d_model, d_model, bias=False)
        self.w_o = Linear(d_model, d_model)
        
        self.dropout = Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.w_o(context)
        return output, attention_weights


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.dropout = Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class TransformerBlock(nn.Module):
    """Single transformer block with pre-layer normalization"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-layer norm with residual connection
        attn_output, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x


class NeuralChatModel(nn.Module):
    """
    Advanced Neural Chat Model - Transformer-based architecture
    Optimized for contextual conversation and text generation
    """
    
    def __init__(self, vocab_size: int, d_model: int = 768, num_heads: int = 12,
                 num_layers: int = 12, d_ff: int = 3072, max_seq_len: int = 2048,
                 dropout: float = 0.1, pad_token_id: int = 0):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        
        # Embedding layers
        self.token_embedding = Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.layer_norm = LayerNorm(d_model)
        self.output_projection = Linear(d_model, vocab_size, bias=False)
        
        # Share weights between input and output embeddings
        self.output_projection.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"Initialized NeuralChatModel with {self.count_parameters():,} parameters")
    
    def _init_weights(self, module):
        """Initialize weights using Xavier/Glorot initialization"""
        if isinstance(module, Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create padding mask for attention"""
        return (x != self.pad_token_id).unsqueeze(1).unsqueeze(2)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal (lower triangular) mask for autoregressive generation"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                return_hidden_states: bool = False) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create masks
        if attention_mask is None:
            attention_mask = self.create_padding_mask(input_ids)
        
        causal_mask = self.create_causal_mask(seq_len, device)
        combined_mask = attention_mask & causal_mask
        
        # Token embeddings + positional encoding
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.position_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Store hidden states if requested
        hidden_states = [] if return_hidden_states else None
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, combined_mask)
            if return_hidden_states:
                hidden_states.append(x)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        # Output projection to vocabulary
        logits = self.output_projection(x)
        
        output = {"logits": logits}
        if return_hidden_states:
            output["hidden_states"] = hidden_states
            
        return output
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 512,
                 temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9,
                 repetition_penalty: float = 1.1, pad_token_id: int = None,
                 eos_token_id: int = None) -> torch.Tensor:
        """
        Generate text using nucleus (top-p) sampling with temperature scaling
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        
        generated = input_ids.clone()
        past_tokens = set()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass
                outputs = self.forward(generated)
                logits = outputs["logits"][:, -1, :]  # Get last token logits
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for token in past_tokens:
                        logits[:, token] /= repetition_penalty
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Add to past tokens for repetition penalty
                past_tokens.add(next_token.item())
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for end of sequence
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break
        
        return generated
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for monitoring"""
        return {
            "model_name": "NeuralChatModel",
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_layers": len(self.transformer_blocks),
            "max_seq_len": self.max_seq_len,
            "total_parameters": self.count_parameters(),
            "device": next(self.parameters()).device.type
        }