import torch
import torch.nn as nn
from tokenizer import get_dataset
from transformer_model import build_transformer
from tqdm import tqdm
from pathlib import Path
import warnings

from torch.utils.tensorboard import SummaryWriter
def train_transformer(src_lang, tgt_lang, seq_len, batch_size, epochs):

    # Set the device to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("In Use: ", device)
    dim_model = 512
    h = 4

    train_loader, val_loader, src_tokenizer, tgt_tokenizer = get_dataset('opus_books', src_lang, tgt_lang, seq_len, batch_size)
    model = build_transformer(dim_model, h, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size(), seq_len, seq_len).to(device)
    writer = SummaryWriter("Exp 1")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-9)
    loss = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id("[PAD]"), label_smoothing=0.1).to(device)    
    global_step = 0
    for epoch in range(epochs):
        
        model.train()
        batch_iterator = tqdm(train_loader, desc=f"Processing:{epoch}")
        for batch in batch_iterator:
            encoder_input = batch["enc_tokens"].to(device)
            decoder_input = batch["dec_tokens"].to(device)
            label = batch["label"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)


            output = model(input = encoder_input, output = decoder_input, encoder_mask = encoder_mask, decoder_mask = decoder_mask)
            l = loss(output.view(-1, tgt_tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{l.item():6.3f}"})
            
            writer.add_scaler("Train Loss", l.item(), global_step)
            writer.flush()

            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step+=1

        model_filename = get_weights_file_path(f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

def get_weights_file_path(epoch: str):
    model_folder = "model_folder"
    model_basename = "model_basename"
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    print("Calling train_transformer()")
    train_transformer("en", "fr", 350, 8, 20)

        



