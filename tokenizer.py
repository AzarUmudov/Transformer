import torch 
import torch.nn as nn
from pathlib import Path
from tokenizers import Tokenizer 
from tokenizers.models import WordPiece, BPE, WordLevel
from tokenizers.trainers import WordPieceTrainer, BpeTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset
from torch.utils.data import random_split, DataLoader
from dataset import TranslationDataset

unk_token = "[UNK]"
special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"]

def load_sentences(dataset, split):
    for data in dataset: 
        yield data['translation'][split]

def load_tokenizer(tokenizer_algo, dataset, split, config):
    if tokenizer_algo == 'BPE':
        tokenizer = Tokenizer(BPE(unk_token=unk_token))
        trainer = BpeTrainer(special_tokens=special_tokens)
    elif tokenizer_algo == 'WordPiece':
        tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
        trainer = WordPieceTrainer(special_tokens=special_tokens)
    elif tokenizer_algo == 'WordLevel':
        tokenizer = Tokenizer(WordLevel(unk_token=unk_token))
        trainer = WordLevelTrainer(special_tokens=special_tokens)
    else:
        raise ValueError("Tokenizer algorithm is not found")
    
    tokenizer_path = Path(config['tokenizer_file'].format(split))
    if not Path.exists(tokenizer_path):
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train_from_iterator(iterator=load_sentences(dataset, split), trainer=trainer)
        tokenizer.save()
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

def get_dataset(config, split_size=0.9):
    dataset = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
    input_tokenizer = load_tokenizer(tokenizer_algo='WordLevel', dataset=dataset, split=config['lang_src'], config=config)
    output_tokenizer = load_tokenizer(tokenizer_algo='WordLevel', dataset=dataset, split=config['lang_tgt'], config=config)

    train_dataset, val_dataset = random_split(dataset=dataset, lengths=[len(dataset)*split_size, len(dataset)*(1-split_size)])
    train_dataset= TranslationDataset(dataset=train_dataset, seq_len=config['seq_len'], tokenizer_input=input_tokenizer,
                       tokenizer_output=output_tokenizer, input_lang=config['lang_src'], output_lang=config['lang_tgt'])
    val_dataset= TranslationDataset(dataset=val_dataset, seq_len=config['seq_len'], tokenizer_input=input_tokenizer,
                       tokenizer_output=output_tokenizer, input_lang=config['lang_src'], output_lang=config['lang_tgt'])
        
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)

    return train_loader, val_loader, input_tokenizer, output_tokenizer

