from torch import nn
from models import Encoder


class MNISTClassifier(nn.Module):
    def __init__(self, encoder: Encoder):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.latent_size = 10  # encoder.latent_size #TODO change back

        self.dense = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size),
            nn.ReLU(),
            nn.Linear(self.latent_size, 10),
            nn.Softmax()
        )

    def forward(self, x):
        z = self.encoder(x)[:,0:10]
        ps = self.dense(z)
        return ps
