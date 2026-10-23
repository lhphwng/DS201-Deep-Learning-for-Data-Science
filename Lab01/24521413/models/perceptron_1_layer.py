import torch
from torch import nn
import torch.nn.functional as F


class Perceptron_1_layer(nn.Module):
    def __init__(self, image_size: tuple, num_labels: int):
        super(). __init__()

        h, w = image_size

        self.linear = nn.Linear(
            in_features = w*h,
            out_features = num_labels
        )

    def forward(self, x: torch.tensor):
        bs = x.shape[0]
        x = x.reshape(bs, -1)

        output = self.linear(x)
        output = F.softmax(output, dim=1)

        return output
