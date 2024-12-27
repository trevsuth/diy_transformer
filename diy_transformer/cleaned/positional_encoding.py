import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq = seq
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq, d_model)
        pe = torch.zeros(seq, d_model)

        # Create a vector of shape (seq)
        position = torch.arange(0, seq, dtype=torch.float).unsqueeze(1)  # (seq, 1)

        # Create a vector of shape (d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_modell / 2)

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(
            position * div_term
        )  # sin(position * 10000 ** (2i / d_model))

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(
            position * div_term
        )  # sin(position * 10000 ** (2i / d_model))

        # Add a batch dimensions to original positional encoding
        pe = pe.unsqueeze(0)  # (1, seq, d_model)

        # Register the positional encoding as a buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(
            False
        )  # (batch, seq, d_model)
        return self.dropout(x)

