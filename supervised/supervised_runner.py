import neptune.new as neptune
import os

import numpy as np
import torch

from supervised.mnistclassifier import MNISTClassifier
from running import get_dataloader
from tqdm import trange, tqdm
from datetime import datetime


def create_classifier_from_neptune(run_name: str) -> (torch.nn.Module, neptune.Run):
    destination_path = f"pretrained_models/{run_name}_model.pt"

    nept_log = neptune.init(project="cj.griffin/beta-vae",
                            api_token=os.getenv('NEPTUNE_API_TOKEN'),
                            run=run_name)
    nept_log["model_checkpoints/model"].download(destination_path)
    nept_log.stop()

    if torch.cuda.is_available():
        vae_model = torch.load(destination_path)
    else:
        vae_model = torch.load(destination_path, map_location=torch.device('cpu'))
    encoder = vae_model.encoder
    classifier = MNISTClassifier(encoder)
    return classifier


def step_classifier(model, loader, optimizer, train=True):
    if train:
        model.train()  # sets the module in training mode.
    else:
        model.eval()

    total_loss = 0.0
    total_n = 0
    device = next(model.parameters()).device

    criterion = torch.nn.CrossEntropyLoss()

    num_correct = 0
    num_total = 0

    for X, y in loader:
        X = X.to(device)

        if train:
            optimizer.zero_grad()

        ps = model(X)

        with torch.no_grad():
            preds = ps.argmax(dim=1)
            corrects = (preds == y)
            num_correct += corrects.sum()
            num_total += len(corrects)

        losses = criterion(ps, y)

        total_loss += losses.sum().item()

        if train:
            batch_loss = losses.mean()
            batch_loss.backward()
            optimizer.step()

        total_n += X.shape[0]

    loss = total_loss / total_n
    accuracy = num_correct / num_total
    return loss, accuracy


def train_supervised(model, optimizer, nept_log, num_samples, batch_size_train, batch_size_test): #, epochs):
    train_data = get_dataloader("MNIST", batch_size=batch_size_train, is_train=True, num_samples=num_samples)
    test_data = get_dataloader("MNIST", batch_size=batch_size_test, is_train=False)
    epochs = int(1e3)
    train_losses = np.empty(epochs)
    test_losses = np.empty(epochs)
    for epoch in trange(epochs):
        train_losses[epoch], _ = step_classifier(model, train_data, optimizer, train=True)
        nept_log["supervised/Train Loss"].log(train_losses[epoch])
        test_losses[epoch], acc = step_classifier(model, test_data, optimizer, train=False)
        nept_log["supervised/Test Loss"].log(test_losses[epoch])
        nept_log["supervised/Test Acc"].log(acc)
        print("epoch {}: Test Loss {}".format(epoch, test_losses[epoch]))


def experiment_supervised(
        original_run_name: str,
        # epochs: int = 1,
        lr: float = 0.01,
        num_samples = int(1e4),
        batch_size_train: int = 100,
        batch_size_test: int = 1000):

    classifier = create_classifier_from_neptune(original_run_name)
    nept_log = neptune.init(project="cj.griffin/beta-vae",
                            api_token=os.getenv('NEPTUNE_API_TOKEN'))
    nept_log["Original"] = original_run_name
    nept_log["num_samples"] = num_samples
    optimizer = torch.optim.Adam(params=classifier.parameters(), lr=lr)
    train_supervised(classifier, optimizer, nept_log, num_samples,
                     batch_size_train=batch_size_train,
                     batch_size_test=batch_size_test)
                     # epochs=epochs)
    torch.save(classifier, "/supervised_checkpoints/classifier.pt")
    nept_log["model_checkpoints/supervised"].upload("supervised_checkpoints/classifier.pt")


if __name__ == "__main__":
    # model, _ = load_model_from_neptune("BVAE-24")
    # import matplotlib.pyplot as plt
    # plt.imshow(model(torch.ones(1,1,28,28))[0].view(28,28).detach().numpy())
    # plt.show()
    for original_run_name in ["BVAE-130", "BVAE-126", "BVAE-133", "BVAE-127", "BVAE-128", ]:
        for num_samples in [10,100,1000,10000]:
            experiment_supervised(original_run_name=original_run_name,
                                  num_samples=num_samples)
