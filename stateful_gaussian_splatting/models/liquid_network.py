"""
Liquid Neural Network implementation for stateful processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple, Union


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for time steps.
    """
    
    def __init__(self, dim: int):
        """
        Initialize the sinusoidal positional embedding.
        
        Args:
            dim: Dimension of the embedding.
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the sinusoidal positional embedding.
        
        Args:
            time: Time steps of shape (batch_size,).
            
        Returns:
            Positional embeddings of shape (batch_size, dim).
        """
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((torch.sin(embeddings), torch.cos(embeddings)), dim=-1)
        
        # If dim is odd, pad with zeros
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1, 0, 0))
            
        return embeddings


class LiquidAttention(nn.Module):
    """
    Multi-head attention mechanism for the Liquid Neural Network.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """
        Initialize the liquid attention module.
        
        Args:
            hidden_size: Size of the hidden dimension.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the liquid attention module.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).
            mask: Optional mask tensor of shape (batch_size, seq_len).
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, and values
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Final projection
        output = self.proj(context)
        
        return output


class LiquidLayer(nn.Module):
    """
    A single layer of the Liquid Neural Network.
    """
    
    def __init__(
        self,
        hidden_size: int,
        activation: str = "swish",
        dropout: float = 0.1,
        use_attention: bool = True,
        attention_heads: int = 4
    ):
        """
        Initialize the liquid layer.
        
        Args:
            hidden_size: Size of the hidden dimension.
            activation: Activation function to use.
            dropout: Dropout probability.
            use_attention: Whether to use attention mechanism.
            attention_heads: Number of attention heads.
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.use_attention = use_attention
        
        # Attention mechanism
        if use_attention:
            self.attention = LiquidAttention(hidden_size, attention_heads, dropout)
            self.layer_norm1 = nn.LayerNorm(hidden_size)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get the activation function."""
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "swish":
            return nn.SiLU()  # SiLU is the same as Swish
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the liquid layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).
            mask: Optional mask tensor of shape (batch_size, seq_len).
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        # Apply attention if enabled
        if self.use_attention:
            attn_output = self.attention(x, mask)
            x = self.layer_norm1(x + self.dropout(attn_output))
        
        # Apply feed-forward network
        ff_output = self.ff(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        
        return x


class LiquidNetwork(nn.Module):
    """
    Liquid Neural Network for stateful processing.
    
    This network incorporates temporal dynamics and stateful processing through
    a combination of attention mechanisms and recurrent connections.
    """
    
    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 4,
        activation: str = "swish",
        dropout: float = 0.1,
        time_embedding_dim: int = 128,
        use_attention: bool = True,
        attention_heads: int = 4,
        device: torch.device = None
    ):
        """
        Initialize the Liquid Neural Network.
        
        Args:
            hidden_size: Size of the hidden dimension.
            num_layers: Number of liquid layers.
            activation: Activation function to use.
            dropout: Dropout probability.
            time_embedding_dim: Dimension of the time embedding.
            use_attention: Whether to use attention mechanism.
            attention_heads: Number of attention heads.
            device: Device to use for computation.
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Liquid layers
        self.layers = nn.ModuleList([
            LiquidLayer(
                hidden_size=hidden_size,
                activation=activation,
                dropout=dropout,
                use_attention=use_attention,
                attention_heads=attention_heads
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        # Initialize parameters
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights of the network."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the Liquid Neural Network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).
            time: Time steps of shape (batch_size,).
            mask: Optional mask tensor of shape (batch_size, seq_len).
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        # Embed time
        time_emb = self.time_embed(time)
        
        # Add time embedding to input
        x = x + time_emb.unsqueeze(1)
        
        # Apply liquid layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final projection
        output = self.output_proj(x)
        
        return output