import pickle as pk
import pandas as pd
from pathlib import Path
import torchtext
from torchtext.datasets import SNLI
from torchtext.vocab import GloVe

def get_dataloaders(batch_size: int, data_path: Path):        
    data_path.mkdir(parents=True, exist_ok=True)
    TEXT = torchtext.data.Field(lower=True, batch_first=True, tokenize="spacy", include_lengths=True)
    LABEL = torchtext.data.Field(sequential=False, unk_token=None)  
    train_data, val_data, test_data= SNLI.splits(text_field=TEXT, label_field=LABEL, root=data_path)
    TEXT.build_vocab(train_data, vectors=GloVe(cache=data_path))
    LABEL.build_vocab(train_data)
    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train_data, val_data, test_data), batch_size=batch_size
    )
    return train_iter, val_iter, test_iter, TEXT.vocab

def accuracy(pred, label):
    return (pred.argmax(dim=1) == label).float().mean().item()

def report_senteval(senteval_results):
    # tasks which do not have accuracy metric
    senteval_results.pop('SICKRelatedness', None)
    senteval_results.pop('STS14', None)
   
    scores = {task: {'metric': results['devacc'],
                     'n_samples': results['ndev']}
              for task, results in senteval_results.items()}
    scores_df = pd.DataFrame.from_dict(data=scores)

    tasks = list(scores_df.columns)
    scores_df['macro'] = [scores_df.loc['metric', tasks].mean(), scores_df.loc['n_samples'].sum()]
    scores_df['micro'] = [(scores_df.loc['metric', tasks] * scores_df.loc['n_samples', tasks]).sum()
                          / scores_df.loc['n_samples', tasks].sum(), scores_df.loc['n_samples'].mean()]
    return scores_df

class SentenceEncoder():
    
    def __init__(self, encoder, vocab):
        self.encoder = encoder
        self.vocab = vocab
    
    def encode(self, sentence):
        tokens = torch.tensor([[self.vocab.stoi[word.lower()] for word in sentence.split()]])
        sent_embed = self.encoder.emb(tokens)
        h0 = torch.randn(2, 1, 2048)
        c0 = torch.randn(2, 1, 2048)
        _, (hidden, _) = self.encoder.lstm(sent_embed, (h0, c0))
        hidden_bi = torch.cat((hidden[0], hidden[1]), dim=1)
        return hidden_bi.detach().numpy().squeeze()
    
    def cosine_similarity(self, sentence1, sentence2):
        u = self.encode(sentence1)
        v = self.encode(sentence2)
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))