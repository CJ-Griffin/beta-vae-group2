import numpy as np
import torch
import torch.cuda
import torch.nn
import torch.optim
import torchvision
from models import AE, VAE, VAE_Loss, AE_Loss, Dense28x28, TanhVAE
from neptune import new as neptune
from tqdm import tqdm

from shape_dataset import ShapesDataset
from src.graphing.visualisation import show_images, visualize_latent_space

"""
This file enables the training of VAEs on MNIST or Shapes.
"""


# Given a dataset name, returns the corresponding Dataloader object
# If num_samples is None, return all of the datapoints, else return only a subset.
#   (this is used for measuring sample efficiency)
def get_dataloader(dataset_name: str,
                   batch_size: int,
                   is_train: bool,
                   num_samples=None) -> torch.utils.data.DataLoader:
    if dataset_name == "MNIST":
        dataset = torchvision.datasets.MNIST('data', train=is_train, download=True,
                                             transform=torchvision.transforms.ToTensor())
        if num_samples is not None:
            dataset = torch.utils.data.Subset(dataset, list(range(num_samples)))
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    elif dataset_name == "Shapes":
        dataset = ShapesDataset()
        if num_samples is not None:
            dataset = torch.utils.data.Subset(dataset, list(range(num_samples)))
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # # DEPRECATED - due to GPU constraints
    # elif dataset_name == "CelebA":
    #     return torch.utils.data.DataLoader(
    #         torchvision.datasets.CelebA('data', train=is_train, download=True,
    #                                     transform=torchvision.transforms.Compose([
    #                                         torchvision.transforms.ToTensor(),
    #                                     ])),
    #         batch_size=batch_size, shuffle=True)
    else:
        return NotImplementedError(f"Can't handle dataset with name {dataset_name}")


# Completes a single epoch of training for the (V)AE, with reconstruction (and KL) loss
def step_autoencoder(model, criterion, loader, optimizer, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_n = 0
    device = next(model.parameters()).device

    for X, _ in (loader):
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


# Given a (V)AE, create a dataloader and train for a number of epochs
def train_AE(model, optimizer, criterion, nept_log, dataset_name: str,
             epochs=3, batch_size_train=100, batch_size_test=1000):
    train_loader = get_dataloader(dataset_name=dataset_name, batch_size=batch_size_train, is_train=True)
    test_loader = get_dataloader(dataset_name=dataset_name, batch_size=batch_size_test, is_train=False)

    train_losses = np.empty(epochs)
    test_losses = np.empty(epochs)

    for epoch in tqdm(range(epochs)):
        train_losses[epoch] = step_autoencoder(model=model, criterion=criterion, loader=train_loader,
                                               optimizer=optimizer)
        nept_log["Train Loss"].log(train_losses[epoch])
        test_losses[epoch] = step_autoencoder(model=model, criterion=criterion, loader=test_loader,
                                              optimizer=optimizer, train=False)
        nept_log["Test Loss"].log(test_losses[epoch])


# Given all the parameters of a single training run, set it up, run it and log the results
def run_experiment(model_name: str,
                   latent_size: int = 10,
                   beta: float = 1.0,
                   lr: float = 0.001,
                   epochs: int = 3,
                   dataset_name: str = "MNIST",
                   is_offline: bool = False,
                   is_beta_normalised=False
                   ):
    # Initialise neptune_log, used to store experiment results online.
    # To view the results, visit here: https://app.neptune.ai/cj.griffin/beta-vae/
    nept_log = neptune.init(project="cj.griffin/beta-vae",
                            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ZjE4NGNlOC0wMmFjLTQxZTEtODg1ZC0xMDRhMTg3YjI2ZjAifQ==",
                            mode=("offline" if is_offline else "async"))
    nept_log["model_name"] = model_name
    nept_log["lr"] = lr
    nept_log["epochs"] = epochs
    nept_log["dataset_name"] = dataset_name
    nept_log["latent_size"] = latent_size
    if is_beta_normalised:
        beta_norm = beta
        nept_log["norm_beta"] = beta_norm
        M = latent_size
        N = 28 * 28
        beta = beta_norm * N / M
    nept_log["beta"] = beta

    # Generate the model
    if model_name == "AE":
        model = AE(latent_size=latent_size)
        criterion = AE_Loss
    elif model_name == "VAE":
        model = VAE(latent_size=latent_size)
        criterion = (lambda model_output, X: VAE_Loss(model_output, X, beta=beta))
    elif model_name == "TanhVAE":
        model = TanhVAE(latent_size=latent_size)
        criterion = (lambda model_output, X: VAE_Loss(model_output, X, beta=beta))
    elif model_name == "Dense":
        model = Dense28x28()
        criterion = AE_Loss
    else:
        raise NotImplementedError(f"model_name={model_name} not implemented")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    # Train the model
    train_AE(model=model,
             optimizer=optimizer,
             criterion=criterion,
             epochs=epochs,
             nept_log=nept_log,
             dataset_name=dataset_name)

    # Generate visualisations of performance and upload to Neptune
    test_examples, _ = next(iter(get_dataloader(batch_size=1000, is_train=False, dataset_name=dataset_name)))

    try:
        fig3 = visualize_latent_space(test_examples, model.encoder)
        nept_log["vis_latent_space"].upload(fig3)
    except Exception as e:
        pass

    test_examples = test_examples[:10, :].to(device)

    fig1 = show_images(test_examples)

    model_out = model(test_examples)
    if isinstance(model_out, tuple) and len(model_out) == 2:
        model_out = model_out[0]
    fig2 = show_images(model_out)

    nept_log["true_images"].upload(fig1)
    nept_log["recon_images"].upload(fig2)

    torch.save(model, "../models/model_checkpoints/temp.pt")
    nept_log["model_checkpoints/model"].upload("../models/model_checkpoints/temp.pt")

    nept_log.stop()


# An example of how to run autoencoder experiments
if __name__ == "__main__":
    for g_beta in [0.01, 0.1, 1.0, 10.0, 100.0]:
        run_experiment(model_name="TanhVAE",
                       latent_size=6,
                       beta=g_beta,
                       lr=0.001,
                       epochs=2,
                       dataset_name="Shapes")
