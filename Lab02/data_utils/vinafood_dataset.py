import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def collate_fn(items: list[dict]) -> dict[str, torch.Tensor]:
    data = {
        "image": np.stack([item["image"] for item in items], axis=0),
        "label": np.stack([item["label"] for item in items], axis=0)
    }

    data = {
        "image": torch.from_numpy(data["image"]),
        "label": torch.from_numpy(data["label"]).long()
    }

    return data


class VinaFood21Dataset(Dataset):
    def __init__(self, data_dir: str, image_size: int = 224, train: bool = False):
        self.image_size = image_size
        self.train = train

        self.classes = sorted(
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        )
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        self._data = []  # [(image_path, label)]
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            for file_name in sorted(os.listdir(class_dir)):
                if file_name.lower().endswith(IMG_EXTS):
                    self._data.append((
                        os.path.join(class_dir, file_name),
                        self.class_to_idx[class_name]
                    ))

        if len(self._data) == 0:
            raise RuntimeError(f"Không tìm thấy ảnh nào trong {data_dir}")

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        path, label = self._data[idx]

        image = Image.open(path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image = np.asarray(image, dtype=np.float32) / 255.0

        if self.train and np.random.rand() < 0.5:
            image = image[:, ::-1, :]

        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)

        return {
            "image": image,
            "label": label
        }
