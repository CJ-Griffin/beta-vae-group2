from running import run_experiment

if __name__ == "__main__":
    for beta in [1.0, 4.0, 16.0]:  # [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
        run_experiment(model_name="VAE",
                       latent_size=10,
                       beta=beta,
                       lr=0.001,
                       epochs=2,
                       dataset_name="MNIST")
