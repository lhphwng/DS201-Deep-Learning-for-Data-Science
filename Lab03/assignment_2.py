from torch import nn

from assignment_1 import run_classification
from data_utils.uit_vsfc import Vocab
from models.classifiers.gru import GruModel


def main(max_epochs: int = 50, patience: int = 5) -> float:
    print("=" * 20, "Bài 2: GRU 5 lớp - UIT-VSFC", "=" * 20)

    def build_model(vocab: Vocab) -> nn.Module:
        return GruModel(
            vocab_size=vocab.len,
            hidden_size=256,
            n_layer=5,
            n_labels=vocab.n_labels,
            padding_idx=vocab.pad_idx
        )

    return run_classification(
        build_model,
        ckpt_name="assignment_2_gru_best.pt",
        max_epochs=max_epochs,
        patience_limit=patience,
    )


if __name__ == "__main__":
    main()
