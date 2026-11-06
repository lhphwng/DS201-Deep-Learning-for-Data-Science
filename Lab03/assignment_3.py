import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.phonert import IGNORE_INDEX, NerVocab, PhoNER, collate_fn, find_file
from metrics import f1_ner
from models.ner.bilstm_encoder import BiLstmNerModel

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
PHONER_DIR = os.path.join(ROOT, "datasets", "PhoNER_COVID19")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def train(
        model: nn.Module,
        dataloader: DataLoader,
        epoch: int,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        clip_norm: float = 5.0
) -> float:
    running_loss = []
    model.train()

    with tqdm(dataloader, desc=f"Epoch {epoch} - Training") as pbar:
        for item in pbar:
            input_ids = item["input_ids"].to(device)
            lengths = item["lengths"]
            labels = item["labels"].to(device)              # (bs, len)

            logits = model(input_ids, lengths)              # (bs, len, n_tags)
            # CrossEntropy cần (N, C) và (N,) -> gộp batch và len
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

            running_loss.append(loss.item())
            pbar.set_postfix({"loss": np.array(running_loss).mean()})
    return float(np.mean(running_loss))


@torch.no_grad()
def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        epoch: int,
        vocab: NerVocab,
        verbose: bool = False
) -> float:
    model.eval()

    gts = []    # list[list[str]]
    pds = []
    with tqdm(dataloader, desc=f"Epoch {epoch} - Evaluating") as pbar:
        for item in pbar:
            input_ids = item["input_ids"].to(device)
            lengths = item["lengths"]
            labels = item["labels"]                         # (bs, len)

            logits = model(input_ids, lengths)              # (bs, len, n_tags)
            predicted = logits.argmax(dim=-1).cpu()         # (bs, len)

            for pred_seq, gt_seq in zip(predicted, labels):
                mask = gt_seq != IGNORE_INDEX               # bỏ vị trí padding
                gts.append(vocab.decode_tags(gt_seq[mask].tolist()))
                pds.append(vocab.decode_tags(pred_seq[mask].tolist()))

    scores = f1_ner(gts, pds)
    print(f"Precision: {scores['precision']:.4f} | Recall: {scores['recall']:.4f} | F1: {scores['f1']:.4f}")
    if verbose:
        for ent_type, s in scores["per_type"].items():
            print(f"  {ent_type:<22} P={s['precision']:.4f} R={s['recall']:.4f} F1={s['f1']:.4f}")
    return scores["f1"]


def main(max_epochs: int = 50, patience: int = 5, batch_size: int = 32, lr: float = 1e-3) -> float:
    print("=" * 20, "Bài 3: Encoder 5 lớp BiLSTM - NER PhoNER", "=" * 20)

    train_path = find_file(PHONER_DIR, "train_word.conll")
    dev_path = find_file(PHONER_DIR, "dev_word.conll")
    test_path = find_file(PHONER_DIR, "test_word.conll")

    print("Loading vocab ...")
    vocab = NerVocab(train_path=train_path)
    print(f"Vocab size: {vocab.len} | Tags: {vocab.n_tags}")

    print("Creating datasets ...")
    train_dataset = PhoNER(train_path, vocab)
    dev_dataset = PhoNER(dev_path, vocab)
    test_dataset = PhoNER(test_path, vocab)

    print("Creating dataloaders ...")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print("Initializing model and utilities ...")
    model = BiLstmNerModel(
        vocab_size=vocab.len,
        hidden_size=256,
        n_layer=5,
        n_tags=vocab.n_tags,
        padding_idx=vocab.pad_idx
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX).to(device)

    ckpt_path = os.path.join(CHECKPOINT_DIR, "assignment_3_bilstm_ner_best.pt")

    epoch = 0
    best_f1 = 0.0
    patience_count = 0
    while epoch < max_epochs:
        epoch += 1
        train(model, train_dataloader, epoch, loss_fn, optimizer)
        f1 = evaluate(model, dev_dataloader, epoch, vocab)

        if f1 > best_f1:
            best_f1 = f1
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best checkpoint to {ckpt_path}")
        else:
            patience_count += 1

        if patience_count == patience:
            print("Patience reached! Training completed")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Best dev F1: {best_f1}")
    test_f1 = evaluate(model, test_dataloader, epoch, vocab, verbose=True)
    print(f"Entity-level F1 on test set: {test_f1}")
    return test_f1


if __name__ == "__main__":
    main()
