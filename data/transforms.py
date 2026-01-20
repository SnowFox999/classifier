from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from config import IMG_SIZE

def get_transforms(size=224):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(
            size=size, scale=(0.8, 1.2), ratio=(0.9, 1.1)
        ),
        transforms.ColorJitter(
            brightness=0.2,
            saturation=0.2,
            contrast=0.2,
            hue=0.2
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_tfms, val_tfms

