import os

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

IGNORE_INDEX = -100


def find_file(root: str, filename: str) -> str:
    for dirpath, _, files in os.walk(root):
        if filename in files:
            return os.path.join(dirpath, filename)
    raise FileNotFoundError(
        f"Không tìm thấy '{filename}' trong '{root}'. "
        f"Hãy tải PhoNER_COVID19 (word-level) và đặt vào thư mục này."
    )


def read_conll(path: str) -> tuple[list[list[str]], list[list[str]]]:
    sentences, tags = [], []
    words, labels = [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sentences.append(words)
                    tags.append(labels)
                    words, labels = [], []
                continue
            parts = line.split()
            words.append(parts[0])
            labels.append(parts[-1])

    if words:
        sentences.append(words)
        tags.append(labels)

    return sentences, tags


def collate_fn(items: list[dict]) -> dict:
    input_ids = [item["input_ids"] for item in items]
    labels = [item["labels"] for item in items]

    lengths = torch.tensor([ids.shape[0] for ids in input_ids], dtype=torch.long)
    max_len = int(lengths.max())

    input_ids = torch.stack([
        F.pad(ids, pad=(0, max_len - ids.shape[0]), mode="constant", value=0)
        for ids in input_ids
    ])
    labels = torch.stack([
        F.pad(lb, pad=(0, max_len - lb.shape[0]), mode="constant", value=IGNORE_INDEX)
        for lb in labels
    ])

    return {
        "input_ids": input_ids,
        "lengths": lengths,
        "labels": labels
    }


class NerVocab:
    def __init__(self, train_path: str):
        self.pad = "<pad>"
        self.unk = "<unk>"

        sentences, tags = read_conll(train_path)

        all_words = set()
        all_tags = set()
        for words, labels in zip(sentences, tags):
            all_words.update(w.lower() for w in words)
            all_tags.update(labels)

        self.w2i = {self.pad: 0, self.unk: 1}
        for idx, word in enumerate(sorted(all_words), start=2):
            self.w2i[word] = idx
        self.i2w = {idx: word for word, idx in self.w2i.items()}
        
        sorted_tags = ["O"] + sorted(t for t in all_tags if t != "O")
        self.t2i = {tag: idx for idx, tag in enumerate(sorted_tags)}
        self.i2t = {idx: tag for tag, idx in self.t2i.items()}

    @property
    def len(self) -> int:
        return len(self.w2i)

    @property
    def n_tags(self) -> int:
        return len(self.t2i)

    @property
    def pad_idx(self) -> int:
        return self.w2i[self.pad]

    def encode_sentence(self, words: list[str]) -> torch.Tensor:
        unk_id = self.w2i[self.unk]
        return torch.tensor([self.w2i.get(w.lower(), unk_id) for w in words], dtype=torch.long)

    def encode_tags(self, tags: list[str]) -> torch.Tensor:
        return torch.tensor([self.t2i[t] for t in tags], dtype=torch.long)

    def decode_tags(self, tag_ids: list[int]) -> list[str]:
        return [self.i2t[idx] for idx in tag_ids]


class PhoNER(Dataset):
    def __init__(self, path: str, vocab: NerVocab):
        super().__init__()
        self.sentences, self.tags = read_conll(path)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.vocab.encode_sentence(self.sentences[idx]),
            "labels": self.vocab.encode_tags(self.tags[idx])
        }
