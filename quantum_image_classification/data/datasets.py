import numpy as np
from torchvision import datasets, transforms
import os

class MNISTDataHandler:
    def __init__(self, data_dir='./data/mnist'):
        """Initialize the MNIST data handler"""
        self.data_dir = data_dir
        os.makedirs(os.path.join(data_dir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'processed'), exist_ok=True)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Download and load MNIST dataset
        self.train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.MNIST(
            data_dir, train=False, download=True, transform=self.transform)

    def get_numpy_data(self):
        """Get MNIST data as numpy arrays"""
        # Convert PyTorch tensors to numpy arrays
        X_train = self.train_dataset.data.numpy() / 255.0
        y_train = self.train_dataset.targets.numpy()
        X_test = self.test_dataset.data.numpy() / 255.0
        y_test = self.test_dataset.targets.numpy()
        
        return X_train, y_train, X_test, y_test
