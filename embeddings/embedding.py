import torch 
import torch.nn as nn
import math

class PositionEncoding(nn.Module):

    def __init__(self, seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        position_encoding = torch.arange(start=0, end=seq_len, dtype=torch.float).unsqueeze(1)/\
            torch.as_tensor([math.pow(10000, 2*i/d_model) for i in range(d_model)])
        
        position_encoding[:, 0::2] = torch.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = torch.cos(position_encoding[:, 1::2])
        self.position_encoding = position_encoding.unsqueeze(0)

        self.register_buffer(name='positional_encoding', tensor=position_encoding)

    def forward(self, x):
        x = x + (self.position_encoding[:, :x.shape[1], :]).requires_grad_(False)
        x = self.dropout(x)
        return x
        
class InputEmbedding(nn.Module):

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        return x
