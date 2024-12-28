import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    """
    This class implements the embedding layer for a Transformer model, as described in
    'Building a Transformer from Scratch: A Step-by-Step Guide'.

    It maps token indices into dense vectors of size `d_model`. The output of this
    embedding is scaled by the square root of `d_model` to aid training stability.

    Args:
        d_model (int): The dimensionality of the embeddings (and the model).
        vocab_size (int): The size of the vocabulary.

    Attributes:
        d_model (int): The embedding dimensionality.
        vocab_size (int): Size of the vocabulary.
        embedding (nn.Embedding): Embedding layer mapping tokens to vectors of size `d_model`.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Forward pass for the InputEmbeddings module.

        Args:
            x (torch.Tensor): Indices of shape (batch, seq) representing tokens.

        Returns:
            torch.Tensor: Embedded tensor of shape (batch, seq, d_model),
            scaled by sqrt(d_model).
        """
        return self.embedding(x) * math.sqrt(self.d_model)
