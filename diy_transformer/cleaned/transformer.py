import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder
from input_embeddings import InputEmbeddings
from positional_encoding import PositionalEncoding
from projection_layer import ProjectionLayer

class Transformer(nn.Module):
    """
    The full Transformer architecture, consisting of an encoder and a decoder, along with
    input embeddings, positional encodings, and a projection layer for final outputs.

    Explained in 'Building a Transformer from Scratch: A Step-by-Step Guide'.

    Args:
        encoder (Encoder): The encoder module of the Transformer.
        decoder (Decoder): The decoder module of the Transformer.
        src_embed (InputEmbeddings): Embedding layer for the source sequence.
        tgt_embed (InputEmbeddings): Embedding layer for the target sequence.
        src_pos (PositionalEncoding): Positional encoding for source.
        tgt_pos (PositionalEncoding): Positional encoding for target.
        projection_layer (ProjectionLayer): Linear layer to map decoder outputs to vocab logits.

    Attributes:
        encoder (Encoder): Transformer encoder.
        decoder (Decoder): Transformer decoder.
        src_embed (InputEmbeddings): Source embedding layer.
        tgt_embed (InputEmbeddings): Target embedding layer.
        src_pos (PositionalEncoding): Source positional encoding.
        tgt_pos (PositionalEncoding): Target positional encoding.
        projection_layer (ProjectionLayer): Output projection layer.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        """
        Encodes the source sequence.

        Args:
            src (torch.Tensor): Source indices, shape (batch, seq).
            src_mask (torch.Tensor): Source mask.

        Returns:
            torch.Tensor: Encoded source representation (batch, seq, d_model).
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        """
        Decodes the target sequence given the encoder output.

        Args:
            encoder_output (torch.Tensor): Encoder outputs (batch, seq, d_model).
            src_mask (torch.Tensor): Source mask.
            tgt (torch.Tensor): Target indices, shape (batch, seq).
            tgt_mask (torch.Tensor): Target mask.

        Returns:
            torch.Tensor: Decoder output (batch, seq, d_model).
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        """
        Projects the decoder output to the vocabulary space.

        Args:
            x (torch.Tensor): Decoder output (batch, seq, d_model).

        Returns:
            torch.Tensor: Logits over the vocabulary (batch, seq, vocab_size).
        """
        return self.projection_layer(x)
