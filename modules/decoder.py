import torch
import torch.nn as nn
from blocks.attention import MultiHeadAttention
from blocks.position_wise_fnn import PositionWiseFNN
from blocks.layer_normalization import LayerNormalization

class DecoderBlock(nn.Module):

    def __init__(self, seq_len: int, multi_head_attention: MultiHeadAttention, cross_head_attention: MultiHeadAttention, feed_forward: PositionWiseFNN) -> None:
        super().__init__()
        self.multi_head_attention = multi_head_attention
        self.cross_head_attention = cross_head_attention
        self.feed_forward = feed_forward
        self.layer_norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.layer_norm(x + self.multi_head_attention(x, x, x, target_mask))
        x = self.layer_norm(x + self.cross_head_attention(x, encoder_output, encoder_output, src_mask))
        x = self.layer_norm(x + self.feed_forward(x))
        return x
    
class Decoder(nn.Module):

    def __init__(self, decoders: nn.ModuleList) -> None:
        super().__init__()
        self.decoders = decoders   

    def forward(self, x, encoder_output, src_mask, target_mask):
        for decoder in self.decoders:
            x = decoder(x, encoder_output, src_mask, target_mask)
        return x
    





