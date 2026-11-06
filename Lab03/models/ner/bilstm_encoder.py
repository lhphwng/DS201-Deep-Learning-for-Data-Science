import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLstmEncoder(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
            n_layer: int,
            padding_idx: int = 0,
            dropout: float = 0.2
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layer,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layer > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)
        self.output_size = hidden_size * 2

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        max_len = inputs.size(1)

        embedded = self.embedding(inputs) 
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_out, _ = self.lstm(packed)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=max_len)

        return self.dropout(outputs)


class BiLstmNerModel(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
            n_layer: int,
            n_tags: int,
            padding_idx: int = 0,
            dropout: float = 0.2
    ):
        super().__init__()

        self.encoder = BiLstmEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_layer=n_layer,
            padding_idx=padding_idx,
            dropout=dropout
        )
        self.classifier = nn.Linear(self.encoder.output_size, n_tags)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(inputs, lengths)
        return self.classifier(feats)
