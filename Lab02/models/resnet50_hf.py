import torch.nn as nn
from transformers import ResNetForImageClassification

HF_MODEL_NAME = "microsoft/resnet-50"


class HFResNet50(nn.Module):
    def __init__(self, num_classes, model_name=HF_MODEL_NAME, class_names=None):
        super().__init__()
        kwargs = {}
        if class_names is not None:
            kwargs["id2label"] = {i: c for i, c in enumerate(class_names)}
            kwargs["label2id"] = {c: i for i, c in enumerate(class_names)}
        self.model = ResNetForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,   # thay lớp phân loại 1000 lớp của ImageNet
            **kwargs,
        )

    def forward(self, x):
        return self.model(pixel_values=x).logits
