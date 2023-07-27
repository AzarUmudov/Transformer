import torch
import torch.nn as nn
from blocks.attention import MultiHeadAttention
from blocks.position_wise_fnn import PositionWiseFNN
from blocks.layer_normalization import LayerNormalization
    
class EncoderBlock(nn.Module):

    def __init__(self, seq_len: int, multi_head_attention: MultiHeadAttention, feed_forward: PositionWiseFNN) -> None:
        super().__init__()
        self.multi_head_attention = multi_head_attention
        self.feed_forward = feed_forward
        self.layer_norm =LayerNormalization()
    
    def forward(self, x, src_mask):
        x = self.layer_norm(x + self.multi_head_attention(x, x, x, src_mask))
        x = self.layer_norm(x + self.feed_forward(x))
        return x
        
class Encoder(nn.Module):

    def __init__(self, encoders: nn.ModuleList) -> None:
        super().__init__()
        self.encoders = encoders

    def forward(self, x, src_mask):
        for encoder in self.encoders:
            x = encoder(x, src_mask)
        return x

