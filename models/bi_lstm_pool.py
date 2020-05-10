import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import matplotlib.pyplot as plt


class BiLSTMPool(nn.Module):
    
    def __init__(self, embeddings, batch_size, hidden_size, device, num_layers=1):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self.lstm = nn.LSTM(bidirectional=True, input_size=embeddings.shape[-1], hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.h0 = torch.randn(num_layers*2, batch_size, hidden_size).to(device)
        self.c0 = torch.randn(num_layers*2, batch_size, hidden_size).to(device)
    
    def forward(self, sentence):
        sentence_embed = self.emb(sentence[0])
        x_packed = pack_padded_sequence(sentence_embed, lengths=sentence[1], batch_first=True, enforce_sorted=False)
        output, (sent_hidden, _) = self.lstm(x_packed, (self.h0, self.c0))
        output, seq_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output[output == 0] = -1e9
        sent_hidden_pooled, _ = torch.max(output, dim=1)
        return sent_hidden_pooled 

    def visualize(self, sentence, vocab):
        tokens = torch.tensor([[vocab.stoi[word.lower()] for word in sentence]])
        sentence_embed = self.emb(tokens)

        h0 = torch.randn(2, 1, 2048)
        c0 = torch.randn(2, 1, 2048)
        outputs = self.lstm(sentence_embed, (h0, c0))[0].squeeze()

        outputs, idxs = torch.max(outputs, dim=0)
        idxs = idxs.detach().numpy()
        argmaxs = [np.sum((idxs==k)) for k in range(len(sentence))]
        
        x = range(tokens.shape[1])
        y = [100.0 * n / np.sum(argmaxs) for n in argmaxs]
        plt.xticks(x, sentence, rotation=45)
        plt.bar(x, y)
        plt.ylabel('%')
        plt.title('Visualisation of words importance')
        plt.show()