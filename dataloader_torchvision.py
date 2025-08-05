import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class PermuteChannel:
    def __call__(self, x):
        return x.permute(1, 2, 0)  # (C, H, W) → (H, W, C)
    
def build_dataloader(data_root, batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=data_root, train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
            PermuteChannel(),
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=data_root, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
            PermuteChannel(),
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True)
    return train_loader, val_loader