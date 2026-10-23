import os

import torch

from data import get_dataloader
from models import Perceptron_1_layer, Perceptron_3_layer
from metrics import calculate_all_metrics


def train(model, train_loader, optimizer, device):

    model.train()

    total_loss = 0

    for batch in train_loader:

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        output = model(images)

        loss = -torch.log(
            output[torch.arange(labels.size(0)), labels] + 1e-8
        ).mean()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, test_loader, device):

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for batch in test_loader:

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            output = model(images)

            predictions = torch.argmax(output, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    return y_true, y_pred


def print_results(results):

    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision Macro: {results['precision_macro']:.4f}")
    print(f"Recall Macro: {results['recall_macro']:.4f}")
    print(f"F1 Macro: {results['f1_macro']:.4f}")

    print("\nMetrics for each digit:")

    for digit in range(10):

        print(
            f"Digit {digit}: "
            f"Precision = {results['precision'][digit]:.4f}, "
            f"Recall = {results['recall'][digit]:.4f}, "
            f"F1 = {results['f1'][digit]:.4f}"
        )


def train_model(
    model,
    train_loader,
    test_loader,
    optimizer,
    epochs,
    device,
    checkpoint_path
):

    for epoch in range(epochs):

        loss = train(
            model,
            train_loader,
            optimizer,
            device
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {loss:.4f}"
        )

    y_true, y_pred = evaluate(
        model,
        test_loader,
        device
    )

    results = calculate_all_metrics(
        y_true,
        y_pred,
        num_classes=10
    )

    print_results(results)

    torch.save(
        model.state_dict(),
        checkpoint_path
    )


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    os.makedirs("checkpoints", exist_ok=True)

    # ==========================================
    # DATA
    # ==========================================

    train_loader = get_dataloader(
        "data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte",
        batch_size=64,
        shuffle=True
    )

    test_loader = get_dataloader(
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte",
        batch_size=64,
        shuffle=False
    )

    # ==========================================
    # BAI 1 - PERCEPTRON 1 LAYER
    # ==========================================

    print("\n")
    print("=" * 50)
    print("BAI 1 - PERCEPTRON 1 LAYER")
    print("=" * 50)

    model_1 = Perceptron_1_layer(
        image_size=(28, 28),
        num_labels=10
    ).to(device)

    optimizer_1 = torch.optim.SGD(
        model_1.parameters(),
        lr=0.01
    )

    train_model(
        model=model_1,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer_1,
        epochs=10,
        device=device,
        checkpoint_path="checkpoints/perceptron_1_layer.pth"
    )

    # ==========================================
    # BAI 2 - PERCEPTRON 3 LAYER
    # ==========================================

    print("\n")
    print("=" * 50)
    print("BAI 2 - PERCEPTRON 3 LAYER")
    print("=" * 50)

    model_3 = Perceptron_3_layer(
        image_size=(28, 28),
        num_labels=10
    ).to(device)

    optimizer_3 = torch.optim.SGD(
        model_3.parameters(),
        lr=0.01
    )

    train_model(
        model=model_3,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer_3,
        epochs=10,
        device=device,
        checkpoint_path="checkpoints/perceptron_3_layer.pth"
    )


if __name__ == "__main__":
    main()