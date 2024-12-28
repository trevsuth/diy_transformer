import torch
import torch.nn as nn
import math

# Import all the classes from their respective files
from input_embeddings import InputEmbeddings
from positional_encoding import PositionalEncoding
from layer_normalization import LayerNormalization
from feed_forward_block import FeedForwardBlock
from multi_head_attention_block import MultiHeadAttentionBlock
from residual_connection import ResidualConnection
from encoder_block import EncoderBlock
from encoder import Encoder
from decoder_block import DecoderBlock
from decoder import Decoder
from projection_layer import ProjectionLayer
from transformer import Transformer

def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq: int,
    tgt_seq: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    """
    Constructs a full Transformer model as described in 'Building a Transformer from Scratch:
    A Step-by-Step Guide'.

    Args:
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        src_seq (int): Maximum sequence length for the source.
        tgt_seq (int): Maximum sequence length for the target.
        d_model (int): Dimensionality of embeddings (and the model). Defaults to 512.
        N (int): Number of encoder and decoder layers. Defaults to 6.
        h (int): Number of attention heads. Defaults to 8.
        dropout (float): Dropout probability. Defaults to 0.1.
        d_ff (int): Dimensionality of the feed-forward network. Defaults to 2048.

    Returns:
        Transformer: A Transformer model instance.
    """
    # Embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Positional encodings
    src_pos = PositionalEncoding(d_model, src_seq, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq, dropout)

    # Encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            d_model, encoder_self_attention_block, feed_forward_block, dropout
        )
        encoder_blocks.append(encoder_block)

    # Decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            d_model,
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    # Encoder and Decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Projection
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Full Transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )

    # Initialize parameters using Xavier uniform
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


if __name__ == "__main__":
    # Example usage (if you want to quickly test instantiation):
    model = build_transformer(
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        src_seq=128,
        tgt_seq=128,
        d_model=512,
        N=6,
        h=8,
        dropout=0.1,
        d_ff=2048,
    )
    print(model)
