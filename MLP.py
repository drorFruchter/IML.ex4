import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import helpers
from helpers import *
import pandas as pd

GREEN = '\033[32m'
BLUE = '\033[34m'
RESET = '\033[0m'

class EuropeDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
        """
        #### YOUR CODE HERE ####

        self.data = pd.read_csv(csv_file)
        self.features = torch.tensor(self.data[['long', 'lat']].values, dtype=torch.float32)
        self.labels = torch.tensor(self.data['country'].values, dtype=torch.long)

        # Load the data into a tensors
        # The features shape is (n,d)
        # The labels shape is (n)
        # The feature dtype is float
        # THe labels dtype is long

        #### END OF YOUR CODE ####


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        #### YOUR CODE HERE ####
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data row
        
        Returns:
            dictionary or list corresponding to a feature tensor and it's corresponding label tensor
        """
        #### YOUR CODE HERE ####
        return self.features[idx], self.labels[idx]
    

class MLP(nn.Module):
    def __init__(self, num_hidden_layers, hidden_dim, output_dim, batch_norm=False):
        super(MLP, self).__init__()
        """
        Args:
            num_hidden_layers (int): The number of hidden layers, in total you'll have an extra layer at the end, from hidden_dim to output_dim
            hidden_dim (int): The hidden layer dimension
            output_dim (int): The output dimension, should match the number of classes in the dataset
        """
        #### YOUR CODE HERE ####
        self.layers = nn.ModuleList([nn.Linear(2, hidden_dim)])
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.batch_norm = batch_norm

    def forward(self, x):
        #### YOUR CODE HERE ####
        for layer in self.layers:
            x = layer(x)
            if self.batch_norm:
                x = nn.BatchNorm1d(x)
            x = self.activation(x)
        x = self.output_layer(x)
        return x


def eval(model, criterion, loader):
    model.eval()
    with torch.no_grad():
        #### YOUR CODE HERE ####
        # perform validation loop and test loop here

        correct = 0
        total = 0
        running_loss = 0.0

        for inputs, labels in loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        model_loss = running_loss / len(loader)
        model_acc = 100 * correct / total

    return model_loss, model_acc


def train(train_dataset, val_dataset, test_dataset, model, lr=0.001, epochs=50, batch_size=256):    

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)    
    
    #### YOUR CODE HERE ####
    # initialize your criterion and optimizer here
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []
    iterations = []


    for ep in range(epochs):
        iterations_per_epoch = 0
        model.train()
        #### YOUR CODE HERE ####
        # perform training epoch here
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            iterations_per_epoch += 1

        train_loss = running_loss / len(trainloader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        iterations.append(iterations_per_epoch)

        val_loss, val_acc = eval(model, criterion, valloader)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        test_loss, test_acc = eval(model, criterion, testloader)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f'Epoch {ep+1} took {GREEN}{iterations_per_epoch}{RESET} iterations, Train Acc: {train_accs[-1]:.3f}, Val Acc: {val_accs[-1]:.3f}, Test Acc: {test_accs[-1]:.3f}')

    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, iterations

def plot(train_data, val_data, test_data, title):
    plt.figure()
    plt.plot(train_data, label='Train', color='red')
    plt.plot(val_data, label='Val', color='blue')
    plt.plot(test_data, label='Test', color='green')
    plt.title(title)
    plt.legend()
    plt.show()

lrs = [1., 0.01, 0.001, 0.00001]
colors = ['red', 'blue', 'green', 'cyan']

batches_sizes = [1, 16, 128, 1024]
epochs = [1, 10, 50, 50]

if __name__ == '__main__':
    # seed for reproducibility
    torch.manual_seed(42)

    train_dataset = EuropeDataset('train.csv')
    val_dataset = EuropeDataset('validation.csv')
    test_dataset = EuropeDataset('test.csv')

    #### YOUR CODE HERE #####
    # Find the number of classes, e.g.:
    output_dim = len(train_dataset.labels.unique())

    default_model = MLP(6, 16, output_dim)

    _, default_train_accs, default_val_accs, default_test_accs, default_train_losses, default_val_losses, default_test_losses, _ =\
        train(train_dataset, val_dataset, test_dataset, default_model, lr=0.001, epochs=50, batch_size=256)

    plot(default_train_losses, default_val_losses, default_test_losses, 'Losses')
    plot(default_train_accs, default_val_accs, default_test_accs, 'Accuracies')

    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')
    plot_decision_boundaries(default_model, test_data[['long', 'lat']].values, test_data['country'].values,
                             'Decision Boundaries', implicit_repr=False)

    #Q6.1.2

    #Q1 - Learning rates
    plt.figure()
    for i, lr in enumerate(lrs):
        model = MLP(num_hidden_layers=6, hidden_dim=16, output_dim=output_dim)
        _, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, _ = train(train_dataset,
                                                                                              val_dataset, test_dataset,
                                                                                              model, lr=lr,
                                                                                             epochs=50, batch_size=256)
        plt.plot(val_losses, label=f'lr = {lr}', color=colors[i])
    plt.title("Validation Loss vs Learning Rate")
    plt.legend()
    plt.show()

    #Q2 - Epochs
    model = MLP(6, 16, output_dim)

    _, _, _, _, train_losses, val_losses, test_losses, _ = train(train_dataset, val_dataset,
                                                                                      test_dataset, model, lr=0.001,
                                                                                      epochs=100, batch_size=256)

    plot(train_losses, val_losses, test_losses, '100 Epochs Losses')

    #Q3 - Batch Norm
    model = MLP(6, 16, output_dim, batch_norm=True)

    _, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, _ = train(train_dataset, val_dataset,
                                                                                      test_dataset, model, lr=0.001,
                                                                                      epochs=50, batch_size=256)
    plot(train_losses, val_losses, test_losses, 'Batch Norm Losses')

    #Q4 - Batch Size
    plt.figure()
    for i in range(len(batches_sizes)):
        batch, epoch = batches_sizes[i], epochs[i]
        model = MLP(6, 16, output_dim)

        _, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, iterations = train(train_dataset, val_dataset,
                                                                                          test_dataset, model, lr=0.001,
                                                                                          epochs=epoch, batch_size=batch)
        plt.plot(val_accs, label=f'Batch Size: {batch}', color=colors[i])

        print(f'speed of model of batch {batch} and epoch {epoch}: {iterations} iterations')

