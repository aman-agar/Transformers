import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TranslatorDataset(Dataset):
    def __init__(self, src_lang, tgt_lang, src_tokenizer, tgt_tokenizer, ds, seq_len) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.ds = ds

        self.sos_token = torch.tensor([tgt_tokenizer.token_to_id("[SOS]")], dtype = torch.int64)
        self.eos_token = torch.tensor([tgt_tokenizer.token_to_id("[EOS]")], dtype = torch.int64)
        self.pad_token = torch.tensor([tgt_tokenizer.token_to_id("[PAD]")], dtype = torch.int64)


    
    def __len__(self):
        return(len(self.ds)) 

    def __getitem__(self, index):
        

        # Get target pair from index
        pair = self.ds[index]
        # Get the src_text and tgt_text from target pair
        src_text = pair['translation'][self.src_lang]
        tgt_text = pair['translation'][self.tgt_lang]
        
        # Tokenize the text
        enc_tokens = self.src_tokenizer.encode(src_text).ids
        dec_tokens = self.tgt_tokenizer.encode(tgt_text).ids
        # calculate number of paddings needed
        enc_paddings = self.seq_len - len(enc_tokens) - 2
        dec_paddings = self.seq_len - len(enc_tokens) - 1
        
        # Add sos, eos and paddings
        enc_tokens = torch.cat([
            self.sos_token,
            torch.tensor(enc_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token * enc_paddings], dtype = torch.int64)
        ], dim=0)
        dec_tokens = torch.cat([
            self.sos_token,
            torch.tensor(dec_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_paddings, dtype = torch.int64)
        ], dim=0)
        label = torch.cat([
            
            torch.tensor(dec_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_paddings, dtype = torch.int64)

        ], dim=0)
        
        # Return a dictionary 
        return {

            "encoder_input": enc_tokens,
            "decoder_input": dec_tokens,
            "src_text": src_text,
            "tgt_text": tgt_text,
            "encoder_mask": (enc_tokens != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (dec_tokens != self.pad_token).unsqueeze(0).int() & casual_mask(dec_tokens.size(0)),
            "label": label
        }

def casual_mask(size):
    mask = torch.triu(torch.ones((1,1,size)), diagonal=1).type(torch.int)
    return mask