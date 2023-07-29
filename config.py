from pathlib import Path 

def get_config(seq_len):
    return {
        "lang": "de-en",
        "lang_src": "en",
        "lang_tgt": "de",
        "batch_size": 8,
        "num_epochs": 10,
        "lr": 10**-4,
        "seq_len": seq_len,
        "d_model": 512,
        "d_ff": 2048,
        "dropout":10**-1,
        "head": 8,
        "num_enc": 6,
        "num_dec": 6,
        "model_folder": "weights",
        "model_filename": "transformer_",
        "tokenizer_file": "tokenizer_{0}.json",
        "results_folder": "runs/transformer" 
    }

def get_weights_path(config, epoch):
    return str(Path('.')/config['model_folder']/f"{config['model_filename']}{epoch}.pt")