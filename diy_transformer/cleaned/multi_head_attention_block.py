import torch
import torch.nn as nn
import math


class MultiHeadAttentionBlock(nn.Module):
    """
    Implements multi-head attention as described in 'Building a Transformer from
    Scratch: A Step-by-Step Guide'. The input is projected into multiple heads, where
    each head performs scaled dot-product attention, and outputs are then concatenated.

    Args:
        d_model (int): Dimensionality of the model.
        h (int): Number of attention heads.
        dropout (float): Dropout probability on attention weights.

    Attributes:
        d_model (int): Dimensionality of the model.
        h (int): Number of heads.
        d_k (int): Dimensionality per head (d_model/h).
        w_q (nn.Linear): Linear projection for query vectors.
        w_k (nn.Linear): Linear projection for key vectors.
        w_v (nn.Linear): Linear projection for value vectors.
        w_o (nn.Linear): Final linear projection after concatenating all heads.
        dropout (nn.Dropout): Dropout layer applied to the attention weights.
    """

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by number of heads h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Static method that computes scaled dot-product attention.

        Args:
            query (torch.Tensor): Query vectors (batch, h, seq, d_k).
            key (torch.Tensor): Key vectors (batch, h, seq, d_k).
            value (torch.Tensor): Value vectors (batch, h, seq, d_k).
            mask (torch.Tensor or None): Optional mask for attention.
            dropout (nn.Dropout): Dropout layer for attention weights.

        Returns:
            torch.Tensor: Output of the attention, shape (batch, h, seq, d_k).
            torch.Tensor: Attention weights, shape (batch, h, seq, seq).
        """
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # large negative value for the masked positions
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        """
        Forward pass for MultiHeadAttentionBlock.

        Args:
            q (torch.Tensor): Query vectors (batch, seq, d_model).
            k (torch.Tensor): Key vectors (batch, seq, d_model).
            v (torch.Tensor): Value vectors (batch, seq, d_model).
            mask (torch.Tensor or None): Optional mask for attention.

        Returns:
            torch.Tensor: Multi-head attention output of shape (batch, seq, d_model).
        """
        # Linear projections
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Reshape into (batch, h, seq, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        # Scaled dot-product attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        # Concatenate and project
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)
