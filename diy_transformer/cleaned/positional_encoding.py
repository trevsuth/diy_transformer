import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    This class adds positional information to the token embeddings using sine and
    cosine functions, as explained in 'Building a Transformer from Scratch: A
    Step-by-Step Guide'.

    Since the Transformer has no inherent notion of sequence order, these positional
    encodings enable the model to learn the positional context of tokens.

    Args:
        d_model (int): The dimensionality of the embeddings (and the model).
        seq (int): The maximum sequence length.
        dropout (float): Dropout probability applied to the sum of token embeddings
            and positional encodings.

    Attributes:
        d_model (int): The embedding dimensionality.
        seq (int): Maximum sequence length.
        dropout (nn.Dropout): Dropout layer to avoid overfitting.
        pe (torch.Tensor): Buffer holding the positional encoding values.
    """

    def __init__(self, d_model: int, seq: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq = seq
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq, d_model)
        pe = torch.zeros(seq, d_model)

        # Create a vector of shape (seq)
        position = torch.arange(0, seq, dtype=torch.float).unsqueeze(1)

        # Create a vector of shape (d_model/2)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension
        pe = pe.unsqueeze(0)  # shape (1, seq, d_model)

        # Register the positional encoding as a buffer, so it is saved in the state_dict
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Forward pass for the PositionalEncoding module.

        Args:
            x (torch.Tensor): Input embeddings of shape (batch, seq, d_model).

        Returns:
            torch.Tensor: The input plus positional encoding, with shape (batch, seq, d_model).
        """
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
