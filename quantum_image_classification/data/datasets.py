from torchvision import datasets, transforms
import numpy as np

class MNISTDataHandler:
    def __init__(self, data_dir='./data/mnist'):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.MNIST(
            data_dir, train=False, download=True, transform=self.transform)

    def get_numpy_data(self):
        X_train = self.train_dataset.data.numpy().reshape(-1, 784) / 255.0
        y_train = self.train_dataset.targets.numpy()
        X_test = self.test_dataset.data.numpy().reshape(-1, 784) / 255.0
        y_test = self.test_dataset.targets.numpy()
        return X_train, y_train, X_test, y_test
