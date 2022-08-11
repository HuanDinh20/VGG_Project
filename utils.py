import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision import transforms


def image_augmentation():
    train_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    test_aug = transforms.Compose([
        transforms.Resize((224, 244)),
        transforms.ToTensor()
    ])
    return train_aug, test_aug


def get_FashionMNIST(root, train_aug, test_aug):
    train_set = FashionMNIST(root=root, train=True, transform=train_aug, download=True)
    test_set = FashionMNIST(root=root, train=True, transform=train_aug, download=True)
    return train_set, test_set


def create_data_loader(train_set, test_set, batch_size):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader, test_loader


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device
