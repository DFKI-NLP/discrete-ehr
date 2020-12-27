import torch
import torch.nn as nn
from torch.nn import functional as F


class Word2Vec(nn.Module):
    def __init__(self, num_embeddings, emb_dim, **kwargs):
        super().__init__()
        self.Wi = nn.Embedding(num_embeddings, emb_dim)
        self.Wo = nn.Linear(emb_dim, num_embeddings, bias=False)

    def forward(self, input):
        emb = self.Wi(input)
        x = self.Wo(emb)
        return x
