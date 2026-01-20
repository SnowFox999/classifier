from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from config import IMG_SIZE

def get_transforms():
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_tfms, val_tfms, weights
