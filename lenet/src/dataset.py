import torch
from torchvision import datasets, transforms

def get_mnist_data(data_dir='./data'):
    """
    Downloads the MNIST dataset and applies transformations.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset

def get_data_loaders(train_dataset, test_dataset, batch_size=64):
    """
    Creates data loaders for the training and testing datasets.
    """
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
