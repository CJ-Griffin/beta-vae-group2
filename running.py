import time

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt


def get_dataloader(batch_size, train):
    return torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data', train=train, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=True)


def step_autoencoder(model, criterion, loader, optimizer, train=True):
    if train:
        model.train()  # sets the module in training mode.
    else:
        model.eval()

    total_loss = 0.0
    total_n = 0
    device = next(model.parameters()).device

    for X, _ in loader:
        X = X.to(device)

        if train:
            optimizer.zero_grad()

        X_out = model(X)

        losses = criterion(X_out, X)

        total_loss += losses.sum().item()

        if train:
            batch_loss = losses.mean()
            batch_loss.backward()
            optimizer.step()

        total_n += X.shape[0]

    loss = total_loss / total_n
    return loss


def train_and_plot(model, optimizer, criterion, epochs=3, batch_size_train=100, batch_size_test=1000):
    start = time.time()

    train_loader = get_dataloader(batch_size=batch_size_train, train=True)
    test_loader = get_dataloader(batch_size=batch_size_test, train=False)

    train_losses = np.empty(epochs)
    test_losses = np.empty(epochs)

    for epoch in range(epochs):
        train_losses[epoch] = step_autoencoder(model=model, criterion=criterion, loader=train_loader,
                                               optimizer=optimizer)
        test_losses[epoch] = step_autoencoder(model=model, criterion=criterion, loader=test_loader,
                                              optimizer=optimizer, train=False)
        print("epoch {}: Test Loss {}".format(epoch, test_losses[epoch]))

    print("time: {}".format(time.time() - start))

    plt.plot(range(len(train_losses)), train_losses, label="Train Loss", )
    plt.plot(range(len(test_losses)), test_losses, label="Test Loss", alpha=0.9)
    plt.legend()
    plt.show()
