import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, latent_size=2):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.latent_size = latent_size
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
        x = self.fc2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_size=2, last_layer="sig"):
        super().__init__()

        self.convT1 = nn.ConvTranspose2d(10, 1, kernel_size=5)
        self.convT2 = nn.ConvTranspose2d(20, 10, kernel_size=5)
        self.fc1 = nn.Linear(50, 320)
        self.fc2 = nn.Linear(latent_size, 50)
        self.last_layer = torch.tanh if last_layer == "tanh" else torch.sigmoid

    def forward(self, x):
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc1(x))
        x = x.view(-1, 20, 4, 4)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.convT2(x)
        x = F.relu(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.convT1(x)
        x = self.last_layer(x)

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
    def __init__(self, latent_size=10, last_layer="sig"):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = Encoder(latent_size * 2)
        self.decoder = Decoder(latent_size, last_layer=last_layer)

    def forward(self, X):
        X = self.encoder(X)
        mu, log_var = torch.split(X, self.latent_size, dim=1)

        Z_0 = torch.randn_like(mu)
        Z = Z_0 * torch.exp(0.5 * log_var) + mu

        X = self.decoder(Z)

        # we need to also return the KL term for the loss:
        KL = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)

        return (X, KL)


class TanhVAE(VAE):
    def __init__(self, latent_size=10):
        super().__init__(latent_size=latent_size, last_layer="tanh")


class Dense28x28(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 784),
            nn.ReLU(),
            nn.Linear(784, 784),
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, im):
        return self.dense(im)


def VAE_Loss(model_output, X, beta=1.0):
    X_out, KL = model_output
    return AE_Loss(X_out, X) + beta * KL


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


class MNISTClassifier(nn.Module):
    def __init__(self, encoder: Encoder):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.latent_size = encoder.latent_size

        self.dense = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size),
            nn.ReLU(),
            nn.Linear(self.latent_size, 10),
            nn.Softmax()
        )

    def forward(self, x):
        z = self.encoder(x)[:, 0:self.latent_size]
        ps = self.dense(z)
        return ps
