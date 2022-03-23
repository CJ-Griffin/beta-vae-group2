import math

import torch
from torch import nn as nn
from torch.nn import functional as F
from torchsummary import summary

def get_conv_out_wh(height, width, num_conv_layers, conv_kernel_size, max_pool_kernel_size):
    for i in range(num_conv_layers):
        height = math.ceil((height - conv_kernel_size + 1) / max_pool_kernel_size)
        width = math.ceil((width - conv_kernel_size + 1) / max_pool_kernel_size)
    return height, width


# from copy_of_encoder_experiments import g_mseloss

class Encoder(nn.Module):
    def __init__(self,
                 image_dim=(1, 28, 28),
                 layer_specs=[(5, 10, 2), (5, 20, 2)],
                 latent_size=2):
        super().__init__()
        self.image_dim = image_dim
        depths = [self.image_dim[0]] + [d for _, d, _ in layer_specs]
        self.layer_specs = layer_specs
        self.conv_layers = nn.ModuleList(
            nn.Conv2d(depths[i], depths[i + 1], layer_specs[i][0])
            for i in range(len(layer_specs))
        )

        self.fc1 = nn.LazyLinear(50)
        self.fc2 = nn.Linear(50, latent_size)

    def conv_forward(self, x):
        for conv, spec in zip(self.conv_layers, self.layer_specs):
            x = conv(x)
            x = F.max_pool2d(x, spec[2])
            x = F.relu(x)
        return x

    def forward(self, x):
        x = self.conv_forward(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

    def get_last_conv_dim(self):
        with torch.no_grad():
            dummy = torch.zeros([1] + list(self.image_dim))
            vec = self.conv_forward(dummy)
            return vec.shape[1:]


class Decoder(nn.Module):
    def __init__(self,
                 last_conv_dim,
                 image_dim=(1, 28, 28),
                 layer_specs=[(5, 10, 2), (5, 20, 2)],
                 latent_size=2):
        super().__init__()
        self.last_conv_dim = last_conv_dim
        self.image_dim = image_dim
        depths = [self.image_dim[0]] + [d for _, d, _ in layer_specs]
        self.layer_specs = layer_specs
        self.conv_layers = nn.ModuleList(
            reversed([nn.ConvTranspose2d(depths[i+1], depths[i], layer_specs[i][0])
            for i in range(len(layer_specs))])
        )
        flat_dim = last_conv_dim[0] * last_conv_dim[1] * last_conv_dim[2]
        self.fc1 = nn.Linear(50, flat_dim)
        self.fc2 = nn.Linear(latent_size, 50)

    def conv_forward(self, x):
        for i, (conv_transp, spec) in enumerate(zip(self.conv_layers, self.layer_specs)):
            x = F.interpolate(x, scale_factor=spec[2], mode='nearest')
            x = conv_transp(x)
            if i == len(self.conv_layers) - 1:
                x = F.sigmoid(x)
            else:
                x = F.relu(x)
        return x

    def forward(self, x):
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc1(x))
        x = x.view(-1, *self.last_conv_dim)
        x = self.conv_forward(x)
        return x


class AE(nn.Module):
    def __init__(self, image_dim, latent_size=10):
        super().__init__()
        self.encoder = Encoder(image_dim=image_dim, latent_size=latent_size)
        last_conv_dim = self.encoder.get_last_conv_dim()
        self.decoder = Decoder(image_dim=image_dim, latent_size=latent_size, last_conv_dim=last_conv_dim)
        self.latent_size = latent_size

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class VAE(nn.Module):
    def __init__(self, image_dim, latent_size=10):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = Encoder(image_dim=image_dim, latent_size=latent_size * 2)
        last_conv_dim = self.encoder.get_last_conv_dim()
        self.decoder = Decoder(image_dim=image_dim,
                               latent_size=latent_size,
                               last_conv_dim=last_conv_dim)

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


if __name__ == "__main__":
    model = VAE(image_dim=(3, 218, 178))
    model(torch.ones((1, 3, 218, 178)))
