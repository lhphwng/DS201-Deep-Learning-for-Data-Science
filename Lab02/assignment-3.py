import os 
os.makedirs("./checkpoints/assignment-3", exist_ok=True)

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

from data_utils.vinafood_dataset import VinaFood21Dataset, collate_fn
from models.resnet18 import ResNet18

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(dataloader: DataLoader, model: nn.Module) -> dict:
    model.eval()
    predictions = []
    trues = []
    with torch.no_grad():
        for item in tqdm(dataloader, desc="Evaluating"):
            image = item["image"].to(device)
            label = item["label"].to(device)
            output = model(image) 
            output = output.argmax(dim=-1)

            predictions.extend(output.cpu().tolist())
            trues.extend(label.cpu().tolist())

    return {
        "accuracy": accuracy_score(trues, predictions),
        "precision": precision_score(trues, predictions, average="macro", zero_division=0),
        "recall": recall_score(trues, predictions, average="macro", zero_division=0),
        "f1": f1_score(trues, predictions, average="macro", zero_division=0)
    }

if __name__ == "__main__":
    train_dataset = VinaFood21Dataset(
        data_dir="./data_utils/VinaFood21/train",
        train=True
    )

    test_dataset = VinaFood21Dataset(
        data_dir="./data_utils/VinaFood21/test",
        train=False
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    model = ResNet18(num_classes=len(train_dataset.classes)).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 20
    best_score = 0
    best_score_name = "f1"
    for epoch in range(EPOCHS):
        losses = []
        model.train()
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1} - Training") as pb:
            for item in pb:
                image = item["image"].to(device)
                label = item["label"].to(device)
                
                # forward pass
                output = model(image)
                loss = loss_fn(output, label)
                losses.append(loss.item())

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pb.set_postfix({
                    "loss": np.array(losses).mean()
                })
        
        scores = evaluate(test_dataloader, model)
        for score in scores:
            print(f"\t- {score}: {scores[score]}")
            
        current_score = scores[best_score_name]
        if current_score > best_score:
            best_score = current_score
            torch.save(
                model.state_dict(),
                "./checkpoints/assignment-3/best_model.pth"
            )
