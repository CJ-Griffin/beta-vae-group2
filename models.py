import torch
from torch import nn as nn
from torch.nn import functional as F


# from copy_of_encoder_experiments import g_mseloss

class Encoder(nn.Module):
    def __init__(self, latent_size=2):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, latent_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_size=2):
        super().__init__()

        self.convT1 = nn.ConvTranspose2d(10, 1, kernel_size=5)
        self.convT2 = nn.ConvTranspose2d(20, 10, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(50, 320)
        self.fc2 = nn.Linear(latent_size, 50)

    def forward(self, x):
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc1(x))
        x = x.view(-1, 20, 4, 4)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.convT2(x)
        x = F.relu(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.convT1(x)
        x = torch.sigmoid(x)

        return x


class AE(nn.Sequential):
    def __init__(self, latent_size=10):
        encoder = Encoder(latent_size=latent_size)
        decoder = Decoder(latent_size=latent_size)
        super().__init__(encoder, decoder)
        self.latent_size = latent_size
        self.encoder = encoder
        self.decoder = decoder


class VAE(nn.Module):
    def __init__(self, latent_size=10):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = Encoder(latent_size * 2)
        self.decoder = Decoder(latent_size)

    def forward(self, X):
        X = self.encoder(X)
        mu, log_var = torch.split(X, self.latent_size, dim=1)

        Z_0 = torch.randn_like(mu)
        Z = Z_0 * torch.exp(0.5 * log_var) + mu

        X = self.decoder(Z)

        # we need to also return the KL term for the loss:
        KL = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)

        # KL = torch.zeros_like(KL) # TODO remove, just seeing what happens

        return (X, KL)


def VAE_Loss(model_output, X, beta=1.0):
    X_out, KL = model_output
    return AE_Loss(X_out, X) + KL


# TODO - is this okay?
g_mseloss = torch.nn.MSELoss(reduction='none')


def AE_Loss(X_out, X):
    losses = g_mseloss(X_out, X)
    losses = losses.view(losses.shape[0], -1)  # flattening losses, keeping batch dimension 0
    losses = losses.sum(dim=1)
    return losses


class VAE_to_encoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.encoder = vae.encoder
        self.latent_size = vae.latent_size

    def forward(self, X):
        X = self.encoder(X)
        mu, log_var = torch.split(X, self.latent_size, dim=1)
        return mu
