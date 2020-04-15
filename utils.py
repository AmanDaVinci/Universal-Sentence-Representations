import torchtext
from torchtext.datasets import SNLI
from torchtext.vocab import GloVe
from pathlib import Path

def get_dataloaders(self, batch_size: int, data_path: Path):        
    data_path.mkdir(parents=True, exist_ok=True)
    TEXT = torchtext.data.Field(lower=True, batch_first=True, tokenize="spacy", include_lengths=True)
    LABEL = torchtext.data.Field(sequential=False, unk_token=None)  
    train_data, val_data, test_data= SNLI.splits(text_field=TEXT, label_field=LABEL, root=data_path)
    TEXT.build_vocab(train_data, vectors=GloVe(cache=data_path))
    LABEL.build_vocab(train_data)
    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train_data, val_data, test_data), batch_size=batch_size
    )
    return train_iter, val_iter, test_iter, TEXT.vocab.vectors

def accuracy(pred, label):
    return (pred.argmax(axis=1) == label).float().mean().item()