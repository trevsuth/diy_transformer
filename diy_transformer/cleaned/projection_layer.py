import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    """
    Maps the decoder's output to the vocabulary logits, as described in
    'Building a Transformer from Scratch: A Step-by-Step Guide'.

    Args:
        d_model (int): Dimensionality of the decoder outputs.
        vocab_size (int): Size of the target vocabulary.

    Attributes:
        proj (nn.Linear): Linear transformation (d_model -> vocab_size).
    """

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        Forward pass for the ProjectionLayer.

        Args:
            x (torch.Tensor): Decoder output of shape (batch, seq, d_model).

        Returns:
            torch.Tensor: Logits over the vocabulary of shape (batch, seq, vocab_size).
        """
        return self.proj(x)
