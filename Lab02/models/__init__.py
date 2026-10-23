from .googlenet import GoogLeNet
from .resnet18 import ResNet18


def get_model(name, num_classes, use_bn=False, class_names=None):
    name = name.lower()
    if name == "googlenet":
        return GoogLeNet(num_classes, use_bn=use_bn)
    if name == "resnet18":
        return ResNet18(num_classes)
    if name == "resnet50":
        from .resnet50_hf import HFResNet50   # import muộn để không bắt buộc cài transformers cho bài 2, 3
        return HFResNet50(num_classes, class_names=class_names)
    raise ValueError(f"Mô hình không hợp lệ: {name} (googlenet | resnet18 | resnet50)")


__all__ = ["GoogLeNet", "ResNet18", "get_model"]
