import neptune.new as neptune

import torch
from matplotlib import pyplot as plt

from src.running import get_dataloader

from visualisation import compare_images


def get_vae_from_neptune(run_name: str) -> (torch.nn.Module, neptune.Run):
    destination_path = f"models/pretrained_models/{run_name}_model.pt"

    nept_log = neptune.init(project="cj.griffin/beta-vae",
                            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ZjE4NGNlOC0wMmFjLTQxZTEtODg1ZC0xMDRhMTg3YjI2ZjAifQ==",
                            run=run_name)
    nept_log["model_checkpoints/model"].download(destination_path)
    beta = nept_log["beta"].fetch()
    beta = float(beta)
    nept_log.stop()

    if torch.cuda.is_available():
        vae_model = torch.load(destination_path)
    else:
        vae_model = torch.load(destination_path, map_location=torch.device('cpu'))
    return vae_model, beta


def get_one_of_each_digit():
    train_data = get_dataloader("MNIST", batch_size=1, is_train=True)
    data_iter = iter(train_data)
    examples = []
    for dig in range(10):
        next_im, next_y = next(data_iter)
        while next_y != dig:
            next_im, next_y = next(data_iter)
        examples.append(next_im)
    return examples


if __name__ == "__main__":
    to_test_2 = list(([
        "BVAE-294",
        "BVAE-295",
        "BVAE-355",
        "BVAE-297",
        "BVAE-298",
        "BVAE-299",
        "BVAE-300"
    ]))
    to_test_10 = list(([
        "BVAE-360",
        "BVAE-302",
        "BVAE-303",
        "BVAE-304",
        "BVAE-305",
        "BVAE-306",
        "BVAE-307"
    ]))
    # model, _ = load_model_from_neptune("BVAE-24")
    # import matplotlib.pyplot as plt
    # plt.imshow(model(torch.ones(1,1,28,28))[0].view(28,28).detach().numpy())
    # plt.show()
    models = {}
    betas = {}
    for original_run_name in to_test_2:  # Change 10 to 2 everywhere to switch
        model, beta = get_vae_from_neptune(run_name=original_run_name)
        models[original_run_name] = model
        betas[original_run_name] = beta
    print(models)

    for fig_no in range(10):
        to_plot_dict = {}
        X_og = get_one_of_each_digit()
        X_og = torch.stack(X_og, dim=0).view(10, 1, 28, 28)
        # print(X_og.shape)
        # fig = show_images(X_og)
        # plt.show()
        # fig.show()
        labeled_dict = {"True": X_og}
        for name in to_test_2:
            X_recon, _ = models[name](X_og)
            label = f"β={betas[name]}"
            labeled_dict[label] = X_recon

        fig = compare_images(labeled_dict)
        plt.savefig(f"images/ls2_{fig_no}.png")
