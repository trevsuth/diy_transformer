import torch
import torch.nn as nn

from multi_head_attention_block import MultiHeadAttentionBlock
from feed_forward_block import FeedForwardBlock
from residual_connection import ResidualConnection

class EncoderBlock(nn.Module):
    """
    A single layer (block) of the Transformer encoder, consisting of:
    1) Multi-head self-attention
    2) Feed-forward network
    3) Residual connections and layer normalization

    Explained in 'Building a Transformer from Scratch: A Step-by-Step Guide'.

    Args:
        features (int): Dimensionality of the embeddings (and the model).
        self_attention_block (MultiHeadAttentionBlock): Self-attention block for this layer.
        feed_forward_block (FeedForwardBlock): Position-wise feed-forward block.
        dropout (float): Dropout probability.

    Attributes:
        self_attention_block (MultiHeadAttentionBlock): Self-attention module.
        feed_forward_block (FeedForwardBlock): Position-wise FFN module.
        residual_connections (nn.Module): Contains two ResidualConnection sub-layers
          for attention and feed-forward network respectively.
    """

    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        """
        Forward pass for the EncoderBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq, d_model).
            src_mask (torch.Tensor or None): Attention mask for padding or future tasks.

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq, d_model),
            after self-attention and feed-forward.
        """
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
