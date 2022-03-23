import os
import time

import numpy as np
import torch
import torch.cuda
import torch.nn
import torch.optim
import torchvision
from models import AE, VAE, VAE_Loss, AE_Loss
from matplotlib import pyplot as plt
from neptune import new as neptune
from tqdm import tqdm

from visualisation import show_images, visualize_latent_space


def get_dataloader(dataset_name: str,
                   batch_size: int,
                   is_train: bool):
    if dataset_name == "MNIST":
        return torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('data', train=is_train, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           # torchvision.transforms.Normalize(
                                           #     (0.1307,), (0.3081,))
                                       ])),
            batch_size=batch_size, shuffle=True)
    elif dataset_name == "CelebA":
        return torch.utils.data.DataLoader(
            torchvision.datasets.CelebA('data', train=is_train, download=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                        ])),
            batch_size=batch_size, shuffle=True)
    else:
        return NotImplementedError(f"Can't handle dataset with name {dataset_name}")


def step_autoencoder(model, criterion, loader, optimizer, train=True):
    if train:
        model.train()  # sets the module in training mode.
    else:
        model.eval()

    total_loss = 0.0
    total_n = 0
    device = next(model.parameters()).device

    for X, _ in tqdm(loader):
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


def train_and_plot(model, optimizer, criterion, nept_log, dataset_name: str,
                   epochs=3, batch_size_train=100, batch_size_test=1000):
    start = time.time()

    train_loader = get_dataloader(dataset_name=dataset_name, batch_size=batch_size_train, is_train=True)
    test_loader = get_dataloader(dataset_name=dataset_name, batch_size=batch_size_test, is_train=False)

    train_losses = np.empty(epochs)
    test_losses = np.empty(epochs)

    for epoch in range(epochs):
        train_losses[epoch] = step_autoencoder(model=model, criterion=criterion, loader=train_loader,
                                               optimizer=optimizer)
        nept_log["Train Loss"].log(train_losses[-1])
        test_losses[epoch] = step_autoencoder(model=model, criterion=criterion, loader=test_loader,
                                              optimizer=optimizer, train=False)
        nept_log["Test Loss"].log(test_losses[-1])
        print("epoch {}: Test Loss {}".format(epoch, test_losses[epoch]))

    print("time: {}".format(time.time() - start))

    # plt.plot(range(len(train_losses)), train_losses, label="Train Loss", )
    # plt.plot(range(len(test_losses)), test_losses, label="Test Loss", alpha=0.9)
    # plt.legend()
    # plt.show()


def run_experiment(model_name: str,
                   latent_size: int = 10,
                   beta: float = 1.0,
                   lr: float = 0.001,
                   epochs: int = 3,
                   dataset_name: str = "MNIST"
                   ):
    nept_log = neptune.init(project="cj.griffin/beta-vae",
                            api_token=os.getenv('NEPTUNE_API_TOKEN'), )
    nept_log["model_name"] = model_name
    nept_log["lr"] = lr
    nept_log["epochs"] = epochs
    nept_log["dataset_name"] = dataset_name
    nept_log["latent_size"] = latent_size
    nept_log["beta"] = beta

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    if model_name == "AE":
        model = AE(latent_size=latent_size)
        criterion = AE_Loss
    elif model_name == "VAE":
        model = VAE(latent_size=latent_size)
        criterion = (lambda model_output, X: VAE_Loss(model_output, X, beta=beta))
    else:
        raise NotImplementedError(f"model_name={model_name} not implemented")
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    train_and_plot(model=model,
                   optimizer=optimizer,
                   criterion=criterion,
                   epochs=epochs,
                   nept_log=nept_log,
                   dataset_name=dataset_name)

    test_examples, _ = next(iter(get_dataloader(batch_size=1000, is_train=False, dataset_name=dataset_name)))

    fig3 = visualize_latent_space(test_examples, model.encoder)
    nept_log["vis_latent_space"].upload(fig3)

    test_examples = test_examples[:10, :].to(device)

    fig1 = show_images(test_examples)
    nept_log["true_images"].upload(fig1)

    model_out = model(test_examples)
    if isinstance(model_out, tuple) and len(model_out) == 2:
        model_out = model_out[0]
    fig2 = show_images(model_out)
    nept_log["recon_images"].upload(fig2)

    torch.save(model, "model_checkpoints/temp.pt")
    nept_log["model_checkpoints/model"].upload("model_checkpoints/temp.pt")

    nept_log.stop()
