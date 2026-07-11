from pathlib import Path

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def _default_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def load_cifar10(data_dir: str = "./data/cifar10") -> tuple[Dataset, Dataset]:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    transform = _default_transform()
    train = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test = CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    return train, test
