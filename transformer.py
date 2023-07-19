import torch
import torch.nn as nn

from encoder import Encoder, EncoderBlock
from decoder import Decoder, DecoderBlock
from embeddings.embedding import PositionEncoding, InputEmbedding
from blocks.attention import MultiHeadAttention
from blocks.position_wise_fnn import PositionWiseFNN

class Transformer(nn.Module):
    def __init__(self, seq_len: int, src_vocab_size: int, trg_vocab_size: int, d_model: int, d_ff: int, h: int, num_encoder: int, num_decoder: int, dropout: float) -> None:
        super().__init__()
        self.enc_input_embedding = InputEmbedding(vocab_size=src_vocab_size, d_model=d_model)
        self.dec_input_embedding = InputEmbedding(vocab_size=trg_vocab_size, d_model=d_model)
        self.position_encoding = PositionEncoding(seq_len=seq_len, d_model=d_model, dropout=dropout)

        encoder_block = EncoderBlock(MultiHeadAttention(d_model=d_model, h=h),
                                    PositionWiseFNN(d_model=d_model, d_ff=d_ff, dropout=dropout))
        self.encoder = Encoder(nn.ModuleList([encoder_block for _ in range(num_encoder)]))

        decoder_block = DecoderBlock(MultiHeadAttention(d_model=d_model, h=h),
                                     MultiHeadAttention(d_model=d_model, h=h),
                                     PositionWiseFNN(d_model=d_model, d_ff=d_ff, dropout=dropout))
        self.decoder = Decoder(nn.ModuleList([decoder_block for _ in range(num_decoder)]))
        self.layer = nn.Linear(in_features=d_model, out_features=trg_vocab_size)
    
    def encode(self, x, src_mask):
        x = self.enc_input_embedding(x)
        x = self.position_encoding(x)
        x = self.encoder(x, src_mask)
        return x
    
    def decode(self, x, encoder_output, src_mask, target_mask):
        x = self.dec_input_embedding(x)
        x = self.position_encoding(x)
        x = self.decoder(x, encoder_output, src_mask, target_mask)
        return x

    def linear_layer(self, x):
        x = self.layer(x)
        x = nn.Softmax(x, dim=-1)
        return x
