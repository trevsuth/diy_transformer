import torch
import torch.nn as nn

from layer_normalization import LayerNormalization


class Decoder(nn.Module):
    """
    The Transformer Decoder, which stacks multiple DecoderBlock layers.

    Args:
        features (int): Dimensionality of the embeddings (and the model).
        layers (nn.ModuleList): A list of DecoderBlock modules.

    Attributes:
        layers (nn.ModuleList): Decoder blocks.
        norm (LayerNormalization): Final layer normalization on the output of the decoder.
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass for the Decoder.

        Args:
            x (torch.Tensor): Embedded target input (batch, seq, d_model).
            encoder_output (torch.Tensor): Encoder outputs (batch, seq, d_model).
            src_mask (torch.Tensor or None): Mask for the source sequence.
            tgt_mask (torch.Tensor or None): Mask for the target sequence.

        Returns:
            torch.Tensor: Decoder output of shape (batch, seq, d_model).
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
