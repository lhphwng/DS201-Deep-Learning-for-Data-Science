import torch
from torch import nn
import torch.nn.functional as F


class Perceptron_3_layer(nn.Module):

    def __init__(
        self,
        image_size: tuple,
        num_labels: int,
        hidden_size_1: int = 128,
        hidden_size_2: int = 64
    ):
        super().__init__()

        h, w = image_size

        self.linear1 = nn.Linear(
            in_features=w * h,
            out_features=hidden_size_1
        )

        self.linear2 = nn.Linear(
            in_features=hidden_size_1,
            out_features=hidden_size_2
        )

        self.linear3 = nn.Linear(
            in_features=hidden_size_2,
            out_features=num_labels
        )

    def forward(self, x: torch.Tensor):

        bs = x.shape[0]

        x = x.reshape(bs, -1)

        output = self.linear1(x)
        output = F.relu(output)

        output = self.linear2(output)
        output = F.relu(output)

        output = self.linear3(output)
        output = F.softmax(output, dim=1)

        return output