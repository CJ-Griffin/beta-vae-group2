import torch
from torch import nn
from torch.nn import functional as F


# Encoder for MNIST or Shapes images
# Takes dim (batch_size, 1, 28, 28)
# Returns dim (batch_size, latent_size)
# If this is used in a VAE, the latent vector will be split in two to give mean and variance
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


# Decoder for MNIST or Shapes images
# Takes dim (batch_size, latent_size)
# Returns dim (batch_size, 1, 28, 28)
# Allows choice of last-layer to enable "tanh" - required for 0-1 normalised datasets
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


# A standard autoencoder for MNIST or Shapes
class AE(nn.Sequential):
    def __init__(self, latent_size=10):
        encoder = Encoder(latent_size=latent_size)
        decoder = Decoder(latent_size=latent_size)
        super().__init__(encoder, decoder)
        self.latent_size = latent_size
        self.encoder = encoder
        self.decoder = decoder


# A VAE autoencoder for MNIST or Shapes
class VAE(nn.Module):
    def __init__(self, latent_size=10, last_layer="sig"):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = Encoder(latent_size * 2)
        self.decoder = Decoder(latent_size, last_layer=last_layer)

    def forward(self, X) -> (torch.Tensor, torch.Tensor):
        X = self.encoder(X)
        mu, log_var = torch.split(X, self.latent_size, dim=1)

        Z_0 = torch.randn_like(mu)
        Z = Z_0 * torch.exp(0.5 * log_var) + mu

        X = self.decoder(Z)

        # we need to also return the KL term for the loss:
        KL = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)

        return X, KL


# Creates a VAE with TanH final activation
# Necessary for images encoded with -ve color values (e.g. normalised Shapes)
class TanhVAE(VAE):
    def __init__(self, latent_size=10):
        super().__init__(latent_size=latent_size, last_layer="tanh")


# A simple network used for debugging
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


# The beta-VAE loss function, scaled by beta
def VAE_Loss(model_output, X, beta=1.0):
    X_out, KL = model_output
    return AE_Loss(X_out, X) + beta * KL


g_mseloss = torch.nn.MSELoss(reduction='none')


# A loss function for standard AutoEncoders
def AE_Loss(X_out, X):
    losses = g_mseloss(X_out, X)
    losses = losses.view(losses.shape[0], -1)  # flattening losses, keeping batch dimension 0
    losses = losses.sum(dim=1)
    return losses


# A classifier using a frozen encoder, used for transfer learning
# It takes an Encoder object, usually taken from a pretrained VAE, and freezes its weights
class MNISTClassifier(nn.Module):
    def __init__(self, encoder: Encoder):
        super().__init__()
        self.encoder = encoder

        # Freeze the weights of the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.latent_size = encoder.latent_size

        # train a dense classifier on top of the encoder
        self.dense = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size),
            nn.ReLU(),
            nn.Linear(self.latent_size, 10),
            nn.Softmax()
        )

    def forward(self, x):
        # We use only [:, 0:self.latent_size] (the mean) as [:, self.latent_size:] contains the variance
        z = self.encoder(x)[:, 0:self.latent_size]
        ps = self.dense(z)
        return ps
