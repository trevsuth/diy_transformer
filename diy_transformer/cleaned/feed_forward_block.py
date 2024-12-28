import torch
import torch.nn as nn


class FeedForwardBlock(nn.Module):
    """
    Implements the position-wise feed-forward network used within the Transformer layers,
    as explained in 'Building a Transformer from Scratch: A Step-by-Step Guide'.

    Args:
        d_model (int): The dimensionality of the embeddings (and the model).
        d_ff (int): The dimensionality of the hidden layer in the feed-forward network.
        dropout (float): Dropout probability to be applied after ReLU activation.

    Attributes:
        linear_1 (nn.Linear): First linear transformation (d_model -> d_ff).
        dropout (nn.Dropout): Dropout layer to reduce overfitting.
        linear_2 (nn.Linear): Second linear transformation (d_ff -> d_model).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        Forward pass for the FeedForwardBlock.

        Args:
            x (torch.Tensor): Tensor of shape (batch, seq, d_model).

        Returns:
            torch.Tensor: Transformed tensor of shape (batch, seq, d_model).
        """
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
