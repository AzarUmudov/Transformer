from pathlib import Path
from tokenizers import Tokenizer 
from tokenizers.models import WordPiece, BPE, WordLevel
from tokenizers.trainers import WordPieceTrainer, BpeTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

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
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer


