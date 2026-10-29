import torch
from torch import nn
from transformers import ResNetForImageClassification


class PretrainedResNet50(nn.Module):
    def __init__(self, num_classes: int = 21, model_name: str = "microsoft/resnet-50"):
        super().__init__()

        self.model = ResNetForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, images: torch.Tensor):

        return self.model(pixel_values=images).logits
