import torch 
import torch.nn as nn
from pathlib import Path
from transformer import Transformer
from processings import dataset
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm 
from config import get_config, get_weights_path
import warnings

def get_model(config, src_len, tgt_len):
    model = Transformer(seq_len=config['seq_len'], src_vocab_size=src_len, trg_vocab_size=tgt_len, d_model=config['d_model'],
                        d_ff=config['d_ff'], h=config['head'], num_encoder=config['num_enc'], num_decoder=config['num_dec'], dropout=config['dropout'])
    return model 

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, input_tokenizer, output_tokenizer = dataset.get_dataset(config=config)

    model = get_model(config, input_tokenizer.get_vocab_size(), output_tokenizer.get_vocab_size()).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'], eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=.1, ignore_index=input_tokenizer.token_to_id('[PAD]')).to(device) 

    # TODO: add preloading option for any error during training phase 
    steps = 0
    writer = SummaryWriter(log_dir=config['results_folder'])
    for epoch in range(config['num_epochs']):
        model.train()
        iterator = tqdm(train_loader, desc=f'Epoch {epoch}: ')
        for batch in iterator:
            optimizer.zero_grad()

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device) 
            label = batch['label'].reshape(-1).to(device)
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
            final_output = model.linear_layer(decoder_output).reshape(-1, output_tokenizer.get_vocab_size())

            loss = loss_fn(final_output, label)
            iterator.set_postfix({"loss": f"{loss.item()}"})

            writer.add_scalar('Training loss', loss.item(), global_step=steps)
            writer.flush() 
            
            loss.backward()
            optimizer.step()
            steps += 1
        
        save_obj = {
            'epoch':epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': optimizer.state_dict(),
            'global_step': steps
        }
        model_filepath = get_weights_path(config)
        torch.save(save_obj, model_filepath)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config(350)
    train(config)
     

