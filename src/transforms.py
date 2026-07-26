"""
Preprocessing + augmentation pipelines, matching Section 3(2) of the paper.

- All images: resize to 224x224 (ImageNet-pretrained ResNet input size),
  cast to tensor, normalize with ImageNet channel mean/std.
- "heavy" augmentation: random rotation, zoom (via RandomResizedCrop),
  horizontal flip, brightness jitter -- applied to train split only.
- "none": just the deterministic resize/normalize, used as the ablation
  control and always used for val/test regardless of the train setting.
"""
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def eval_transform() -> transforms.Compose:
    """Deterministic preprocessing used for val/test, and for train in the
    'no augmentation' ablation arm."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def heavy_train_transform() -> transforms.Compose:
    """Heavy augmentation arm: rotation, zoom/crop, horizontal flip,
    brightness jitter, then the same normalization as eval."""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),  # zoom
        transforms.RandomRotation(degrees=20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_train_transform(augment: bool) -> transforms.Compose:
    return heavy_train_transform() if augment else eval_transform()
