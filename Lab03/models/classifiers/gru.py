import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class GruModel(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
            n_layer: int,
            n_labels: int,
            padding_idx: int = 0,
            dropout: float = 0.2
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=padding_idx
        )

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=n_layer,
            dropout=dropout if n_layer > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            in_features=hidden_size,
            out_features=n_labels
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor):
        embedded_feats = self.embedding(inputs)
        packed = pack_padded_sequence(embedded_feats, lengths.cpu(), batch_first=True, enforce_sorted=False)

        _, features = self.gru(packed)

        features = self.dropout(features[-1])
        logits = self.classifier(features)

        return logits
