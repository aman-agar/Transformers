from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from dataset import TranslatorDataset
from torch.utils.data import random_split, DataLoader

unk_token = "[UNK]"
special_tokens = ["[UNK]", "[SOS]", "[EOS]", "[PAD]"]

def set_tokenizer():
    '''Define the Tokenizer, Trainer and Pre-tokenizer'''
    
    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    trainer = BpeTrainer(special_tokens=special_tokens)
    
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer, trainer

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]
def train_tokenizer(ds, lang):
    tokenizer, trainer = set_tokenizer()
    tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
    tokenizer.save(f'BPE-Tokenizer-{lang}.json')

def get_tokenizer(tk_path, ds ,lang):
    '''Call this method to train or get a trained tokenizer'''
    if not Path.exists(Path(tk_path)):
        train_tokenizer(ds, lang)
    print(tk_path)
    tokenizer = Tokenizer.from_file(tk_path)
    return tokenizer

def get_dataset(path, src_lang, tgt_lang, seq_len, batch_size):
    '''Call this method to get the dataset in form of train loader and val loader'''

    ds = load_dataset(path, f"{src_lang}-{tgt_lang}", split = "train")

    src_tokenizer = get_tokenizer(f"BPE-Tokenizer-{src_lang}.json", ds, src_lang)
    tgt_tokenizer = get_tokenizer(f"BPE-Tokenizer-{tgt_lang}.json", ds, tgt_lang)

    train_size = int(0.9* len(ds))
    val_size = len(ds) - train_size
    train, val = random_split(ds, [train_size, val_size])

    train_data = TranslatorDataset(src_lang, tgt_lang, src_tokenizer, tgt_tokenizer, train, seq_len)
    val_data = TranslatorDataset(src_lang, tgt_lang, src_tokenizer, tgt_tokenizer, val, seq_len)

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size, shuffle=True)

    return train_loader, val_loader, src_tokenizer, tgt_tokenizer

# Downloads the data, tokenize using BPE Tokenizer and save it locally
train_loader, val_loader, src_tokenizer, tgt_tokenizer = get_dataset('opus_books', "en", "fr", 350, 8)
