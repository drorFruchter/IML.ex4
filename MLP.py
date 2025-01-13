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
        self.batch_norm = batch_norm
        self.layers = nn.ModuleList([nn.Linear(2, hidden_dim)])
        if self.batch_norm:
            self.batch_layers = nn.ModuleList([nn.BatchNorm1d(hidden_dim)])

        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            if self.batch_norm:
                self.batch_layers.append(nn.BatchNorm1d(hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        #### YOUR CODE HERE ####
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.batch_norm:
                x = self.batch_layers[i](x)
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
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        model_loss = running_loss / total
        model_acc = 100 * correct / total

    return model_loss, model_acc


def train(train_dataset, val_dataset, test_dataset, model, lr=0.001, epochs=50, batch_size=256, layers_to_monitor=None):

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)    
    
    #### YOUR CODE HERE ####
    # initialize your criterion and optimizer here
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if layers_to_monitor is not None:
        gradient_magnitudes = {layer: [] for layer in layers_to_monitor}

    batches_losses = []
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

        if layers_to_monitor is not None:
            epoch_gradient_magnitudes = {layer: [] for layer in layers_to_monitor}

        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if layers_to_monitor is not None:
                epoch_gradient_magnitudes = batch_grad_magnitude(epoch_gradient_magnitudes, layers_to_monitor, model)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            iterations_per_epoch += 1

            batches_losses.append(loss.item())

        if layers_to_monitor is not None:
            for layer in layers_to_monitor:
                mean_grad_magnitude = np.mean(epoch_gradient_magnitudes[layer])
                gradient_magnitudes[layer].append(mean_grad_magnitude)

        train_loss = running_loss / total_train
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

    if layers_to_monitor is not None:
        return gradient_magnitudes
    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, iterations, batches_losses


def batch_grad_magnitude(epoch_gradient_magnitudes, layers_to_monitor, model):
    for i, layer in enumerate(model.layers):
        if i in layers_to_monitor:
            grad_magnitude = torch.norm(layer.weight.grad, p=2).item() ** 2  # L2 norm squared
            epoch_gradient_magnitudes[i].append(grad_magnitude)
    return epoch_gradient_magnitudes

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
    np.random.seed(42)

    train_dataset = EuropeDataset('train.csv')
    val_dataset = EuropeDataset('validation.csv')
    test_dataset = EuropeDataset('test.csv')

    #### YOUR CODE HERE #####
    # Find the number of classes, e.g.:
    output_dim = len(train_dataset.labels.unique())

    if(input("run default?").lower() == "y"):
        print("running default MLP")
        default_model = MLP(6, 16, output_dim)

        _, default_train_accs, default_val_accs, default_test_accs, default_train_losses, default_val_losses, default_test_losses, _, _ =\
            train(train_dataset, val_dataset, test_dataset, default_model, lr=0.001, epochs=50, batch_size=256)

        plot(default_train_losses, default_val_losses, default_test_losses, 'Losses')
        plot(default_train_accs, default_val_accs, default_test_accs, 'Accuracies')

        plot_decision_boundaries(default_model, test_dataset.data[['long', 'lat']].values, test_dataset.data['country'].values, 'Decision Boundaries', implicit_repr=False)

        print("*" * 30)

    #Q6.1.2

    while(True):
        question = input("enter question: ")
        if question == 'exit':
            break
        elif question == '1':

            #Q1 - Learning rates
            plt.figure()
            for i, lr in enumerate(lrs):
                print(f'mlp with learning rate: {lr}')
                model = MLP(num_hidden_layers=6, hidden_dim=16, output_dim=output_dim)
                _, _, _, _, _, val_losses, _, _, _ = train(train_dataset, val_dataset, test_dataset, model, lr=lr,
                                                                                                     epochs=50, batch_size=256)
                plt.plot(val_losses, label=f'lr = {lr}', color=colors[i])
                print("*" * 15)
            plt.title("Validation Loss vs Learning Rate")
            plt.legend()
            plt.show()

            print("*" * 30)

        elif question == '2':
            #Q2 - Epochs
            print("running MLP with 100 epochs")
            model = MLP(6, 16, output_dim)

            _, _, _, _, train_losses, val_losses, test_losses, _, _ = train(train_dataset, val_dataset,
                                                                                              test_dataset, model, lr=0.001,
                                                                                              epochs=100, batch_size=256)

            plot(train_losses, val_losses, test_losses, '100 Epochs Losses')

            print("*" * 30)

        elif question == '3':
            #Q3 - Batch Norm
            print("running MLP with Batch Normalization")
            model = MLP(6, 16, output_dim, batch_norm=True)

            _, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, _, _ = train(train_dataset, val_dataset,
                                                                                              test_dataset, model, lr=0.001,
                                                                                              epochs=50, batch_size=256)
            plot(train_losses, val_losses, test_losses, 'Batch Norm Losses')

            print("*" * 30)

        elif question == '4':

            #Q4 - Batch Size
            acc_plot = plt.figure()
            batch_loss = plt.figure()
            for i in range(len(batches_sizes)):
                print(f"running MLP with Batch Size: {batches_sizes[i]}")
                batch, epoch = batches_sizes[i], epochs[i]
                model = MLP(6, 16, output_dim)

                _, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, iterations, batch_losses = train(train_dataset, val_dataset,
                                                                                                  test_dataset, model, lr=0.001,
                                                                                                  epochs=epoch, batch_size=batch)
                plt.figure(acc_plot.number)
                plt.plot(val_accs, label=f'Batch Size: {batch}', color=colors[i])

                plt.figure(batch_loss.number)
                plt.plot(batch_losses, label=f'Batch Loss: {batch}', color=colors[i])

                print(f'speed of model of batch {batch} and epoch {epoch}: {np.mean(iterations)} iterations')

                print(f'stability of model of batch {batch} and epoch {epoch}: {np.mean(batch_losses)}')
                print("*" * 15)

            plt.figure(acc_plot.number)
            plt.xlabel('Epoch')
            plt.ylabel('Validation Accuracy')
            plt.title('Validation Accuracy vs. Epoch for Different Batch Sizes')
            plt.legend()
            plt.grid(True)

            plt.figure(batch_loss.number)
            plt.xlabel('Batch Number')
            plt.ylabel('Training Loss (Average per Batch)')
            plt.title('Training Loss vs. Batch for Different Batch Sizes')
            plt.legend()
            plt.grid(True)

            plt.show()

            print("*" * 30)

    #Q2.2.1

    while True:
        question = input("enter question: ")
        if question == 'exit':
            break
        if question == '1':
            # Q1
            depth_width_combinations = [(1, 16), (2, 16), (6, 16), (10, 16), (6, 8), (6, 32), (6, 64)]

            # Train all models
            models = []
            models_train_accuracies = []
            models_val_accuracies = []
            models_test_accuracies = []

            for depth, width in depth_width_combinations:
                print(f'Training model with depth={depth}, width={width}...')
                model = MLP(depth, width, output_dim)
                model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, iterations, batches_losses = train(
                    train_dataset, val_dataset, test_dataset, model, lr=0.001, epochs=50, batch_size=256)
                models.append(model)
                models_train_accuracies.append(train_accs)
                models_val_accuracies.append(val_accs)
                models_test_accuracies.append(test_accs)
                print("*" * 15)

            val_accuracies = [val_accs[-1] for val_accs in
                              models_val_accuracies]  # Validation accuracy at the last epoch
            best_model_idx = np.argmax(val_accuracies)
            worst_model_idx = np.argmin(val_accuracies)

            best_model = models[best_model_idx]
            worst_model = models[worst_model_idx]

            helpers.plot_decision_boundaries(best_model, test_dataset.data[['long', 'lat']].values,
                                             test_dataset.data['country'].values, 'Best Model Decision Boundaries')
            helpers.plot_decision_boundaries(worst_model, test_dataset.data[['long', 'lat']].values,
                                             test_dataset.data['country'].values, 'Worst Model Decision Boundaries')

            print(f"best models paramters: {best_model}")
            print("*" * 30)

            print("plotting accuracy vs depth for width:16...")
            # Plot accuracy vs. depth (for width=16)
            depths = [1, 2, 6, 10]
            train_accs_depth = [models_train_accuracies[i][-1] for i in range(4)]
            val_accs_depth = [models_val_accuracies[i][-1] for i in range(4)]
            test_accs_depth = [models_test_accuracies[i][-1] for i in range(4)]

            plt.figure()
            plt.plot(depths, train_accs_depth, label='Train', marker='o')
            plt.plot(depths, val_accs_depth, label='Validation', marker='o')
            plt.plot(depths, test_accs_depth, label='Test', marker='o')
            plt.xlabel('Depth (Number of Hidden Layers)')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs. Depth (Width=16)')
            plt.legend()
            plt.grid(True)
            plt.show()

            print("plotting accuracy vs width for depth:8...")
            # Plot accuracy vs. width (for depth=6)
            widths = [8, 16, 32]
            train_accs_width = [models_train_accuracies[i][-1] for i in [4, 2, 5]]
            val_accs_width = [models_val_accuracies[i][-1] for i in [4, 2, 5]]
            test_accs_width = [models_test_accuracies[i][-1] for i in [4, 2, 5]]

            plt.figure()
            plt.plot(widths, train_accs_width, label='Train', marker='o')
            plt.plot(widths, val_accs_width, label='Validation', marker='o')
            plt.plot(widths, test_accs_width, label='Test', marker='o')
            plt.xlabel('Width (Number of Neurons per Layer)')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs. Width (Depth=6)')
            plt.legend()
            plt.grid(True)
            plt.show()

            print("*" * 30)

        if question == '2':
            print("running MLP with Gradient Monitoring...")
            model = MLP(100, 4, output_dim)
            gradient_magnitudes = train(train_dataset, val_dataset, test_dataset, model, epochs=10, layers_to_monitor=[0, 30, 60, 90, 95, 99])
            plt.figure()
            for layer, magnitudes in gradient_magnitudes.items():
                plt.plot(magnitudes, label=f'Layer {layer}')
            plt.xlabel('Epoch')
            plt.ylabel('Mean Gradient Magnitude')
            plt.title('Mean Gradient Magnitude vs. Epoch for Selected Layers')
            plt.legend()
            plt.grid(True)
            plt.show()