from supervised_runner import experiment_supervised

if __name__ == "__main__":
    # model, _ = load_model_from_neptune("BVAE-24")
    # import matplotlib.pyplot as plt
    # plt.imshow(model(torch.ones(1,1,28,28))[0].view(28,28).detach().numpy())
    # plt.show()
    for original_run_name in ["BVAE-130"]:  # , "BVAE-126", "BVAE-133", "BVAE-127", "BVAE-128", ]:
        for num_samples in [10]:  # ,100,1000,10000]:
            experiment_supervised(original_run_name=original_run_name,
                                  num_samples=num_samples,
                                  epochs=1)
