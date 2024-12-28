import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    """
    Implements layer normalization as described in 'Building a Transformer from
    Scratch: A Step-by-Step Guide'. It normalizes the activations across the feature
    dimension, helping to stabilize training.

    Args:
        features (int): The dimension of features to be normalized, typically d_model.
        eps (float): A small value to avoid division by zero during normalization.

    Attributes:
        eps (float): Small constant for numerical stability.
        alpha (torch.nn.Parameter): Learnable scale parameter (multiplicative).
        bias (torch.nn.Parameter): Learnable bias parameter (additive).
    """

    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        """
        Forward pass for LayerNormalization.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq, features).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
