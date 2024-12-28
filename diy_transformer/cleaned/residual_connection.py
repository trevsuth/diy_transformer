import torch
import torch.nn as nn
from layer_normalization import LayerNormalization


class ResidualConnection(nn.Module):
    """
    Implements the residual connection pattern used in Transformers, along with
    layer normalization. Described in 'Building a Transformer from Scratch:
    A Step-by-Step Guide'.

    Args:
        features (int): Dimensionality for layer normalization, typically d_model.
        dropout (float): Dropout probability.

    Attributes:
        dropout (nn.Dropout): Dropout layer.
        norm (LayerNormalization): Layer normalization module.
    """

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        """
        Forward pass for ResidualConnection.

        Args:
            x (torch.Tensor): Input tensor (batch, seq, d_model).
            sublayer (Callable): A sub-layer function that takes and returns a tensor.

        Returns:
            torch.Tensor: Tensor after applying layer normalization, sublayer,
            dropout, and the residual connection.
        """
        return x + self.dropout(sublayer(self.norm(x)))
