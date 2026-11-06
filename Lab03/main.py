"""Chạy toàn bộ project.

    python main.py                   # chạy cả Bài 1, 2, 3
    python main.py --tasks 2 3       # chỉ chạy Bài 2 và Bài 3
    python main.py --max_epochs 20   # giới hạn số epoch tối đa
"""
import argparse

import assignment_1
import assignment_2
import assignment_3

TASKS = {
    1: assignment_1.main,   # LSTM  - UIT-VSFC
    2: assignment_2.main,   # GRU   - UIT-VSFC
    3: assignment_3.main,   # BiLSTM encoder - PhoNER (NER)
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3],
                        help="Các bài cần chạy (mặc định: 1 2 3)")
    parser.add_argument("--max_epochs", type=int, default=50, help="Số epoch tối đa mỗi bài")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results = {}
    for task in args.tasks:
        results[task] = TASKS[task](max_epochs=args.max_epochs, patience=args.patience)

    print("\n" + "=" * 20, "TỔNG KẾT (F1 trên tập test)", "=" * 20)
    names = {1: "LSTM (UIT-VSFC)", 2: "GRU (UIT-VSFC)", 3: "BiLSTM Encoder (PhoNER, entity-level)"}
    for task, f1 in results.items():
        print(f"Bài {task} - {names[task]}: {f1:.4f}")
