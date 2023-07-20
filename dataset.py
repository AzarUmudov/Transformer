import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Any

class TranslationDataset(Dataset):

    def __init__(self, dataset, seq_len, tokenizer_input, tokenizer_output, input_lang, output_lang) -> None:
        super().__init__() 
        self.dataset = dataset
        self.seq_len = seq_len
        self.tokenizer_input = tokenizer_input
        self.tokenizer_output = tokenizer_output
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.start_token = torch.Tensor(self.tokenizer_input.token_to_id(['[SOS]']), dtype=torch.int64)
        self.end_token = torch.Tensor(self.tokenizer_input.token_to_id(['[EOS]']), dtype=torch.int64)
        self.pad_token = torch.Tensor(self.tokenizer_input.token_to_id(['[PAD]']), dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Any:
        datapair = self.dataset[index]
        input_text = datapair['translation'][self.input_lang]
        output_text = datapair['translation'][self.output_lang]
        input_ids = self.tokenizer_input.encode(input_text).ids
        output_ids = self.tokenizer_output.encode(output_text).ids
        decoder_pad_tokens = self.seq_len - len(output_ids) - 1
        encoder_pad_tokens = self.seq_len - len(input_ids) - 2
        
        if not (encoder_pad_tokens > 0 and decoder_pad_tokens > 0):
            raise ValueError('Sentence length is not correct')
        
        encoder_input = torch.cat([
            self.start_token,
            torch.Tensor(input_ids, dtype=torch.int64),
            self.end_token,
            torch.Tensor([self.pad_token]*encoder_pad_tokens, dtype=torch.int64)
        ])

        decoder_input = torch.cat([
            self.start_token,
            torch.Tensor(output_ids, dtype=torch.int64),
            torch.Tensor([self.pad_token]*decoder_pad_tokens, dtype=torch.int64)
        ])

        label = torch.cat([
            torch.Tensor(output_ids, dtype=torch.int64),
            self.end_token,
            torch.Tensor([self.pad_token]*decoder_pad_tokens, dtype=torch.int64)
        ])
    
        diagonal_mask = torch.triu(torch.ones(1, decoder_input.size(0), decoder_input.size(0)), diagonal=1).int() == 0
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & diagonal_mask

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "input_text": input_text,
            "output_text": output_text,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask 
        }
