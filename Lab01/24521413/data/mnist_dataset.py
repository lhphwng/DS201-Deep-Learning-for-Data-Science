import torch
from torch.utils.data import Dataset, DataLoader
import idx2numpy
import numpy as np


def collate_fn(items: list[dict]) -> dict[str, torch.Tensor]:

    images = np.stack(
        [np.expand_dims(item["image"], axis=0) for item in items],
        axis=0
    ).astype(np.float32) / 255.0

    labels = np.stack(
        [item["label"] for item in items],
        axis=0
    ).astype(np.int64)

    data = {
        "image": torch.from_numpy(images).float(),
        "label": torch.from_numpy(labels).long()
    }

    return data


class MNISTDataset(Dataset):

    def __init__(self, image_path: str, label_path: str):

        images = idx2numpy.convert_from_file(image_path)
        labels = idx2numpy.convert_from_file(label_path)

        self._data = [
            {
                "image": np.array(image),
                "label": label
            }
            for image, label in zip(images.tolist(), labels.tolist())
        ]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]


def get_dataloader(
    image_path: str,
    label_path: str,
    batch_size: int = 64,
    shuffle: bool = False
):

    dataset = MNISTDataset(
        image_path=image_path,
        label_path=label_path
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

    return dataloader