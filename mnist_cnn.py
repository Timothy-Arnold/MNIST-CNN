import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.utils.data import random_split 
from torchvision import datasets, transforms
import time
import numpy as np

np.random.seed(1)
torch.manual_seed(1)

device = "cuda"

transform=transforms.ToTensor() 

# Split MNIST dataset into train and test
train_set = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_set, validation_set = random_split(train_set, [int(len(train_set) * 5/6), len(train_set) - int(len(train_set) * 5/6)])
test_set = datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64)

class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) 
        self.conv2 = nn.Conv2d(10, 25, kernel_size=5, padding=2)
        self.conv2_drop = nn.Dropout2d(p=0.05)
        self.fc1 = nn.Linear(900, 100)
        self.fc2 = nn.Linear(100, 10)

        # Apply He initialization
        init.kaiming_uniform_(self.fc1.weight)
        init.kaiming_uniform_(self.fc2.weight)

        # Batch Normalization
        # self.bn1 = nn.BatchNorm1d(100)

    def forward(self, x):
        x = F.elu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.elu(F.max_pool2d(self.conv2_drop(self.conv2(x)), kernel_size=2, stride=2))
        x = x.view(-1, 900)
        x = F.elu(self.fc1(x))
        # x = self.bn1(x)
        x = F.dropout(x, training=self.training, p=0.4)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
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
        print(f"Epoch {epoch + 1}  -  Train accuracy: {format(train_accuracy, '.2f')}%  -  Validation accuracy: {format(validation_accuracy, '.2f')}%")

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

    max_epochs = 50
    early_stopping_steps = 5

    start_time = time.time()

    model = MNISTCNN().to(device)
    train(model, train_loader, max_epochs, early_stopping_steps)
    torch.save(model.state_dict(), "trained_models/mnist_cnn_1.pth")

    end_time = time.time()
    time_taken = end_time - start_time

    print(f"Time taken: {np.round(time_taken, 1)}s")

    test_accuracy = test_accuracy(model, test_loader)

    print("-" * 60)
    print('Test accuracy: ', format(test_accuracy, '.2f'), "%")
    print("-" * 60)