import neptune.new as neptune
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from src.graphing.visualisation import compare_images

NUM_GVS = 1


# Fetch the trained VAE model from neptune,
# Then extract and return the decoder with some information
def get_decoder_and_info_from_neptune(run_name: str) -> (torch.nn.Module, dict):
    destination_path = f"../models/pretrained_models/{run_name}_model.pt"

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
    decoder = vae_model.decoder
    return decoder, info


def get_image(z: torch.Tensor, decoder) -> np.ndarray:
    x = z.float()
    # t_out = decoder(z.float())
    # Run this manually to avoid bugs
    x = F.relu(decoder.fc2(x))
    x = F.relu(decoder.fc1(x))
    x = x.view(-1, 20, 4, 4)

    x = F.interpolate(x, scale_factor=2, mode='nearest')
    x = decoder.convT2(x)
    x = F.relu(x)

    x = F.interpolate(x, scale_factor=2, mode='nearest')
    x = decoder.convT1(x)
    x = torch.sigmoid(x)
    return x


def compare_latent_space_dims(run_names: list[str]):
    decoder_infos = [get_decoder_and_info_from_neptune(run_name) for run_name in run_names]
    a_s = [0.1, 0.5, 1]
    b_s = [-0.1, -0.5, -1, 0.1, 0.5, 1]
    im_dict = {}
    for a in a_s:
        for b in b_s:
            for decoder, info in decoder_infos:
                base = torch.tensor([a, b], dtype=float)
                zs = torch.stack([base * i for i in range(-5, 6)], dim=0)
                print(zs.shape)
                images = get_image(zs, decoder)
                im_dict[info["beta"]] = images
            fig = compare_images(im_dict)
            fig.savefig(f"../images/ls_vis_{a, b}.jpg")


RUN_LABELS = [f"BVAE-{i}" for i in [294, 295, 355, 297, 298, 299, 300]]  # 537)]

if __name__ == "__main__":
    compare_latent_space_dims(RUN_LABELS[1:4])
