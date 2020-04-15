import torch
import torch.nn as nn

class EmbeddingEncoder(nn.Module):
    
    def __init__(self, embeddings):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(embeddings, freeze=False)
    
    def forward(self, sentence):
        sentence_embed = self.emb(sentence[0])
        return torch.mean(sentence_embed, dim=1)