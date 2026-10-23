import argparse
import copy
import os
import time
import traceback

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from metrics import compute_metrics
from models import get_model

ALL_MODELS = ["googlenet", "resnet18", "resnet50"]

DEFAULTS = {
    "googlenet": {"epochs": 10, "batch_size": 64, "lr": 1e-4},
    "resnet18":  {"epochs": 10, "batch_size": 64, "lr": 1e-4},
    "resnet50":  {"epochs": 10, "batch_size": 32, "lr": 1e-4},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all", choices=["all"] + ALL_MODELS,
                   help="mặc định 'all': chạy lần lượt cả 3 mô hình")
    p.add_argument("--data_dir", default="./VinaFood21", help="thư mục chứa train/ và test/")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--use_bn", action="store_true", help="thêm BatchNorm cho GoogLeNet")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_loaders(name, args, batch_size):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if name == "resnet50":
        first = transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0))
    else:
        first = transforms.Resize((args.img_size, args.img_size))

    train_tf = transforms.Compose([first, transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_tf = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                                  transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_set = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_tf)
    test_set = datasets.ImageFolder(os.path.join(args.data_dir, "test"), transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    return train_set, test_set, train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    y_true, y_pred, total_loss = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if criterion is not None:
            total_loss += criterion(out, y).item() * x.size(0)
        y_pred.extend(out.argmax(1).cpu().numpy())
        y_true.extend(y.cpu().numpy())
    m = compute_metrics(y_true, y_pred)
    m["loss"] = total_loss / len(loader.dataset)
    return m


def run(name, args, device):
    cfg = dict(DEFAULTS[name])
    for k in ("epochs", "batch_size", "lr"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)

    print("\n" + "=" * 70)
    print(f"MÔ HÌNH: {name} | epochs={cfg['epochs']} | batch_size={cfg['batch_size']} | lr={cfg['lr']}")
    print("=" * 70)
    torch.manual_seed(args.seed)

    train_set, test_set, train_loader, test_loader = build_loaders(name, args, cfg["batch_size"])
    num_classes = len(train_set.classes)
    print(f"Số lớp: {num_classes} | train: {len(train_set)} | test: {len(test_set)}")

    model = get_model(name, num_classes, use_bn=args.use_bn, class_names=train_set.classes).to(device)
    print("Số tham số: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, f"{name}_best.pth")
    best_f1, best_state = -1.0, None

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_loader.dataset)

        m = evaluate(model, test_loader, device, criterion)
        print(f"Epoch {ep:02d}/{cfg['epochs']} | train_loss {train_loss:.4f} | val_loss {m['loss']:.4f} | "
              f"acc {m['accuracy']:.4f} | P {m['precision']:.4f} | R {m['recall']:.4f} | "
              f"F1 {m['f1']:.4f} | {time.time() - t0:.0f}s")

        if m["f1"] > best_f1:
            best_f1, best_state = m["f1"], copy.deepcopy(model.state_dict())
            torch.save({"model": name, "state_dict": best_state,
                        "classes": train_set.classes, "f1": best_f1}, ckpt_path)
            print(f"  -> Lưu checkpoint tốt nhất: {ckpt_path}")

    model.load_state_dict(best_state)
    final = evaluate(model, test_loader, device)
    print(f"\n=== Kết quả cuối {name} (macro) ===")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"{k:>9}: {final[k]:.4f}")
    return final


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    names = ALL_MODELS if args.model == "all" else [args.model]
    results = {}
    for name in names:
        try:
            results[name] = run(name, args, device)
        except Exception:                      # một mô hình lỗi không làm dừng các mô hình còn lại
            print(f"\n[LỖI] {name} không chạy được:")
            traceback.print_exc()
            results[name] = None

    print("\n" + "=" * 70)
    print("TỔNG KẾT (macro, trên tập test)")
    print("=" * 70)
    print(f"{'model':<12}{'accuracy':>10}{'precision':>11}{'recall':>9}{'f1':>9}")
    for name, r in results.items():
        if r is None:
            print(f"{name:<12}{'(lỗi)':>10}")
        else:
            print(f"{name:<12}{r['accuracy']:>10.4f}{r['precision']:>11.4f}{r['recall']:>9.4f}{r['f1']:>9.4f}")


if __name__ == "__main__":
    main()