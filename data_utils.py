
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


def get_mnist_dataloader(batch_size, data_path='./dataset/'):
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将像素值归一化到 [0, 1]
    ])

    train_dataset = datasets.MNIST(
        root=data_path, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_path, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_cifar10_dataloader(batch_size, data_path='./dataset/cifar10/'):
    # ToTensor 把数值归一化到[0, 1]之间
    # transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
    # out = \frac{x - \mu}{\sigma}, 参数分别是三个通道的均值和方差，这样把x由[0, 1]映射到[-0.5, 0.5]之间，均值为0，加速收敛'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
    ])
    training_data = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    validation_data = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

    data_variance = np.var(training_data.data / 255.0)
    print("training_data varianc %.6f" % data_variance)
    training_loader = DataLoader(training_data, batch_size=batch_size,  shuffle=True, pin_memory=True)
    validation_loader = DataLoader(validation_data, batch_size=32, shuffle=True, pin_memory=True)
    return training_loader, validation_loader, data_variance
