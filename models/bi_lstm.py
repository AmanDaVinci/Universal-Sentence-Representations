import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTM(nn.Module):
    
    def __init__(self, embeddings, batch_size, hidden_size=100, num_layers=1):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self.lstm = nn.LSTM(bidirectional=True, input_size=embeddings.shape[-1], hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.h0 = torch.randn(num_layers*2, batch_size, hidden_size)
        self.c0 = torch.randn(num_layers*2, batch_size, hidden_size)
    
    def forward(self, sentence):
        sentence_embed = self.emb(sentence[0])
        x_packed = pack_padded_sequence(sentence_embed, lengths=sentence[1], batch_first=True, enforce_sorted=False)
        _, (sent_hidden, _) = lstm(x_packed, (self.h0, self.c0))
        sent_bi = torch.cat((sent_hidden[0], sent_hidden[1]), dim=1)
        return sent_bi.squeeze() 