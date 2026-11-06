import os
from typing import Callable

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.uit_vsfc import UIT_VSFC, Vocab, collate_fn
from metrics import f1_classification
from models.classifiers.lstm import LstmModel

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
VSFC_DIR = os.path.join(ROOT, "datasets", "UIT-VSFC")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def train(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer
) -> float:
    running_loss = []
    model.train()

    with tqdm(dataloader, desc=f"Epoch {epoch} - Training") as pbar:
        for item in pbar:
            input_ids = item["input_ids"].to(device)
            lengths = item["lengths"]
            labels = item["label"].to(device)

            # forward pass
            logits = model(input_ids, lengths)      # (bs, n_labels)
            loss = loss_fn(logits, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            pbar.set_postfix({"loss": np.array(running_loss).mean()})
    return float(np.mean(running_loss))


@torch.no_grad()
def evaluate(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        epoch: int
) -> float:
    model.eval()

    gts = []        # ground truths
    pds = []        # predictions
    with tqdm(dataloader, desc=f"Epoch {epoch} - Evaluating") as pbar:
        for item in pbar:
            input_ids = item["input_ids"].to(device)
            lengths = item["lengths"]
            labels = item["label"].to(device)       # (bs, )

            logits: torch.Tensor = model(input_ids, lengths)
            predicted_labels = logits.argmax(dim=-1)    # (bs, )

            gts.extend(labels.tolist())
            pds.extend(predicted_labels.tolist())

    f1 = f1_classification(gts, pds, average="macro")
    print(f"F1: {f1}")
    return f1


def run_classification(
        build_model: Callable[[Vocab], nn.Module],
        ckpt_name: str,
        max_epochs: int = 50,
        patience_limit: int = 5,
        batch_size: int = 32,
        lr: float = 1e-3,
        label_key: str = "topic",
) -> float:
    """Huấn luyện + đánh giá một mô hình phân loại trên UIT-VSFC.

    Dùng chung cho Bài 1 (LSTM) và Bài 2 (GRU).
    Trả về F1 trên tập test (dùng checkpoint tốt nhất trên tập dev).
    """
    train_path = os.path.join(VSFC_DIR, "UIT-VSFC-train.json")
    dev_path = os.path.join(VSFC_DIR, "UIT-VSFC-dev.json")
    test_path = os.path.join(VSFC_DIR, "UIT-VSFC-test.json")

    print("Loading vocab ...")
    vocab = Vocab(train_path=train_path, label_key=label_key)

    print("Creating datasets ...")
    train_dataset = UIT_VSFC(path=train_path, vocab=vocab)
    dev_dataset = UIT_VSFC(path=dev_path, vocab=vocab)
    test_dataset = UIT_VSFC(path=test_path, vocab=vocab)

    print("Creating dataloaders ...")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print("Initializing model and utilities ...")
    model = build_model(vocab).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss().to(device)

    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)

    epoch = 0
    best_f1 = 0.0
    patience = 0
    while epoch < max_epochs:
        epoch += 1
        train(model, train_dataloader, epoch, loss_fn, optimizer)
        f1 = evaluate(model, dev_dataloader, epoch)

        if f1 > best_f1:
            best_f1 = f1
            patience = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best checkpoint to {ckpt_path}")
        else:
            patience += 1

        if patience == patience_limit:
            print("Patience reached! Training completed")
            break

    # đánh giá trên test bằng mô hình tốt nhất trên dev
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_f1 = evaluate(model, test_dataloader, epoch)
    print(f"Best dev F1: {best_f1}")
    print(f"F1-score on test set: {test_f1}")
    return test_f1


def main(max_epochs: int = 50, patience: int = 5) -> float:
    print("=" * 20, "Bài 1: LSTM 5 lớp - UIT-VSFC", "=" * 20)

    def build_model(vocab: Vocab) -> nn.Module:
        return LstmModel(
            vocab_size=vocab.len,
            hidden_size=256,
            n_layer=5,
            n_labels=vocab.n_labels,
            padding_idx=vocab.pad_idx
        )

    return run_classification(
        build_model,
        ckpt_name="assignment_1_lstm_best.pt",
        max_epochs=max_epochs,
        patience_limit=patience,
    )


if __name__ == "__main__":
    main()
