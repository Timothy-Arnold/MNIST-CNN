import time
import numpy as np
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# Fix randomness
seed = 123
np.random.seed(seed)
random.seed(123)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda"

transform = transforms.ToTensor()

train_set_full = datasets.MNIST(
    root="data", train=True, download=True, transform=transform
)
labels = [datapoint[1] for datapoint in train_set_full]
train_indices, val_indices = train_test_split(
    list(range(len(train_set_full))), test_size=1 / 6, stratify=labels, random_state=42
)

train_set = Subset(train_set_full, train_indices)
validation_set = Subset(train_set_full, val_indices)
test_set = datasets.MNIST(root="data", train=False, download=True, transform=transform)


# Ensure consistent shuffling
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(42)

train_loader = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
    num_workers=0,
)
validation_loader = DataLoader(
    validation_set,
    batch_size=64,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
    num_workers=0,
)
test_loader = DataLoader(test_set, batch_size=64)


class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()

        self.elu = nn.ELU()
        self.relu = nn.ReLU()

        # Make convolutional layers
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 32, kernel_size=5, padding=2),
            nn.Dropout2d(p=0.05),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.final_grid_dim = 6
        self.hidden_layer_dim = 128

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * (self.final_grid_dim**2), self.hidden_layer_dim)
        self.fc2 = nn.Linear(self.hidden_layer_dim, 10)
        self.softmax = nn.LogSoftmax(dim=1)

        # Apply He initialization
        init.kaiming_uniform_(self.fc1.weight, nonlinearity="leaky_relu")
        init.kaiming_uniform_(self.fc2.weight, nonlinearity="leaky_relu")

        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(self.hidden_layer_dim)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.elu(self.fc1(x))
        x = self.bn1(x)
        x = F.dropout(x, training=self.training, p=0.4)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


def test_accuracy(model, loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    accuracy_percentage = np.round(accuracy * 100, 2)
    return accuracy_percentage


def train(model, train_loader, max_epochs, early_stopping_steps):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())

    validation_decrease_counter = 0
    highest_validation_accuracy = 0
    stopped_early = False

    for epoch in range(max_epochs):
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

        train_accuracy = test_accuracy(model, train_loader)
        validation_accuracy = test_accuracy(model, validation_loader)
        print(f"Epoch {epoch + 1} - Train accuracy: {format(train_accuracy, '.2f')}% - Validation accuracy: {format(validation_accuracy, '.2f')}%")

        if validation_accuracy > highest_validation_accuracy:
            highest_validation_accuracy = validation_accuracy
            validation_decrease_counter = 0
        else:
            validation_decrease_counter += 1

        if validation_decrease_counter == early_stopping_steps:
            print("Early stopping criteria met")
            stopped_early = True
            break

    if not stopped_early:
        print("Maximum number of epochs reached")


if __name__ == "__main__":
    max_epochs = 30
    early_stopping_steps = 5

    start_time = time.time()

    model = MNISTCNN().to(device)
    train(model, train_loader, max_epochs, early_stopping_steps)
    torch.save(model.state_dict(), "trained_models/mnist_cnn.pth")

    end_time = time.time()
    time_taken = end_time - start_time

    print(f"Time taken: {np.round(time_taken, 1)}s")

    test_accuracy = test_accuracy(model, test_loader)
    print("-" * 60)
    print("Test accuracy: ", format(test_accuracy, ".2f"), "%")
    print("-" * 60)
