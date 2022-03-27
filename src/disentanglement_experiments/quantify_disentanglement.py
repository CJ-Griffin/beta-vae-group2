import neptune.new as neptune

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.shape_dataset import get_im_pairs_tensor, ZbDatset

NUM_GVS = 1


def get_encoder_and_info_from_neptune(run_name: str) -> (torch.nn.Module, dict):
    destination_path = f"models/pretrained_models/{run_name}_model.pt"

    nept_log = neptune.init(project="cj.griffin/beta-vae",
                            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ZjE4NGNlOC0wMmFjLTQxZTEtODg1ZC0xMDRhMTg3YjI2ZjAifQ==",
                            run=run_name)
    nept_log["model_checkpoints/model"].download(destination_path)
    info = {"Original": run_name}
    info["latent_size"] = nept_log["latent_size"].fetch()
    try:
        info["beta"] = nept_log["norm_beta"].fetch()
    except Exception as e:
        info["beta"] = nept_log["beta"].fetch()
    nept_log.stop()

    if torch.cuda.is_available():
        vae_model = torch.load(destination_path)
    else:
        vae_model = torch.load(destination_path, map_location=torch.device('cpu'))
    encoder = vae_model.encoder
    return encoder, info


def get_disentanglement_score(train_loader: DataLoader, test_loader: DataLoader):
    classifier = nn.Sequential(
        nn.LazyLinear(4),
        nn.Softmax()
    )
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(classifier.parameters(), lr=0.001)

    epoch_losses = []
    for epoch_no in tqdm(range(1000)):
        epoch_loss = 0
        for z, y_ind in train_loader:
            optimiser.zero_grad()
            pred = classifier(z)
            loss = criterion(pred, y_ind)
            loss.backward()
            epoch_loss += loss.detach()
            optimiser.step()
        epoch_losses.append(epoch_loss)

    # plt.plot(epoch_losses)
    # plt.show()

    corrects = 0
    totals = 0
    with torch.no_grad():
        for z, y_ind in test_loader:
            pred: torch.Tensor = classifier(z)
            pred_c = pred.argmax(dim=1)
            corrects += (pred_c == y_ind).sum()
            totals += len(y_ind)
    return corrects / totals


def compare_disentanglement(run_names: list[str], B=10000, L=25):
    train_ims, train_ys = get_im_pairs_tensor(B, L)
    test_ims, test_ys = get_im_pairs_tensor(1000, L)

    scores = {}

    for run_name in tqdm(run_names):
        encoder, info = get_encoder_and_info_from_neptune(run_name)

        train_ds = ZbDatset(encoder, train_ims, train_ys)
        train_loader = DataLoader(train_ds, batch_size=100)

        test_ds = ZbDatset(encoder, test_ims, test_ys)
        test_loader = DataLoader(test_ds, batch_size=100)

        score = get_disentanglement_score(train_loader, test_loader)
        scores[(info["Original"], info["beta"], info["latent_size"])] = score

    print(scores)


RUN_LABELS = ['BVAE-483', 'BVAE-481', 'BVAE-480', 'BVAE-479', 'BVAE-478',
              'BVAE-477', 'BVAE-476', 'BVAE-475', 'BVAE-474', 'BVAE-473',
              'BVAE-472', 'BVAE-471', 'BVAE-470', 'BVAE-469', 'BVAE-468',
              'BVAE-467', 'BVAE-466', 'BVAE-465', 'BVAE-464', 'BVAE-463',
              'BVAE-462', 'BVAE-461', 'BVAE-460', 'BVAE-459', 'BVAE-458']

if __name__ == "__main__":
    compare_disentanglement(RUN_LABELS, B=10, L=25)
