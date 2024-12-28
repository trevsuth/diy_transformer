import torch
import torch.nn as nn

from layer_normalization import LayerNormalization

class Encoder(nn.Module):
    """
    The Transformer Encoder, which stacks multiple EncoderBlock layers.

    Args:
        features (int): Dimensionality of the embeddings (and the model).
        layers (nn.ModuleList): A list of EncoderBlock modules.

    Attributes:
        layers (nn.ModuleList): Encoder blocks.
        norm (LayerNormalization): Final layer normalization on the output of the encoder.
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        """
        Forward pass for the Encoder.

        Args:
            x (torch.Tensor): Embedded input of shape (batch, seq, d_model).
            mask (torch.Tensor or None): Source mask for self-attention.

        Returns:
            torch.Tensor: Encoder output of shape (batch, seq, d_model).
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
