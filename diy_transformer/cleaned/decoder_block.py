import torch
import torch.nn as nn

from multi_head_attention_block import MultiHeadAttentionBlock
from feed_forward_block import FeedForwardBlock
from residual_connection import ResidualConnection

class DecoderBlock(nn.Module):
    """
    A single layer (block) of the Transformer decoder, consisting of:
    1) Masked multi-head self-attention
    2) Multi-head cross-attention (attending over encoder output)
    3) Feed-forward network
    4) Residual connections and layer normalization

    Explained in 'Building a Transformer from Scratch: A Step-by-Step Guide'.

    Args:
        features (int): Dimensionality of the embeddings (and the model).
        self_attention_block (MultiHeadAttentionBlock): Self-attention block (masked).
        cross_attention_block (MultiHeadAttentionBlock): Cross-attention block (encoder-decoder).
        feed_forward_block (FeedForwardBlock): Position-wise feed-forward block.
        dropout (float): Dropout probability.

    Attributes:
        self_attention_block (MultiHeadAttentionBlock): Masked self-attention.
        cross_attention_block (MultiHeadAttentionBlock): Cross-attention using encoder output.
        feed_forward_block (FeedForwardBlock): Position-wise FFN.
        residual_connections (nn.ModuleList): Three sub-layers for residual connections.
    """

    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass for the DecoderBlock.

        Args:
            x (torch.Tensor): Input tensor from the previous decoder layer (batch, seq, d_model).
            encoder_output (torch.Tensor): Output from the encoder (batch, seq, d_model).
            src_mask (torch.Tensor or None): Source mask (for cross-attention).
            tgt_mask (torch.Tensor or None): Target mask (for self-attention).

        Returns:
            torch.Tensor: Decoder output of shape (batch, seq, d_model).
        """
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        x = self.residual_connections[1](
            x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
