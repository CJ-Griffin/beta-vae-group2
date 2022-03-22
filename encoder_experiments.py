# -*- coding: utf-8 -*-
"""Copy of encoder experiments.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CATb5j96C6GPpqoPKDFO86SZ6kIn7BU1
"""

from models import AE, VAE, VAE_Loss, AE_Loss
from running import run_experiment

if __name__ == "__main__":
    ae = AE()
    run_experiment(ae, AE_Loss, epochs=1)

    vae = VAE()
    run_experiment(vae, VAE_Loss, epochs=1)

    # g_encoder.to(g_device)
    # g_decoder.to(g_device)
    # g_tocanvas = To_Canvas_Format().to(g_device)
    # g_decoder_export = nn.Sequential(g_decoder, g_tocanvas)
    #
    # g_dummy_input, g_dummy_input_y = next(iter(g_test_loader))
    # g_dummy_input = g_dummy_input.to(g_device)
    #
    # g_dummy_latent = g_encoder(g_dummy_input)
    # g_dummy_output = g_decoder_export(g_dummy_latent)
    #
    # export_onnx(model=g_encoder, dummy_input=g_dummy_input, file_name="encoder.onnx")
    # export_onnx(model=g_decoder_export, dummy_input=g_dummy_latent, file_name="decoder.onnx")
    #
    # # output some example data as json
    #
    # export_data_to_json(g_test_loader, 5000, "mnist")

    # g_vae = VAE()
    # g_vae.to(device)
    # g_vae_optimizer = torch.optim.Adam(params=g_vae.parameters(), lr=0.001)
    #
    # train_and_plot(model=g_vae, optimizer=g_vae_optimizer, epochs=3, criterion=VAE_Loss)
    # train_and_plot(model=g_vae, optimizer=g_vae_optimizer, epochs=5, criterion=VAE_Loss)
    # train_and_plot(model=g_vae, optimizer=g_vae_optimizer, epochs=5, criterion=VAE_Loss)
    # train_and_plot(model=g_vae, optimizer=g_vae_optimizer, epochs=5, criterion=VAE_Loss)
    #
    # g_test_examples, _ = next(iter(g_test_loader))
    # g_test_examples = g_test_examples[:10, :].to(g_device)
    #
    # show_images(g_test_examples)
    # g_vae_X_out, _ = g_vae(g_test_examples)
    # show_images(g_vae_X_out)
    #
    # g_vae_optimizer.lr = 0.0001
    # train_and_plot(model=g_vae, optimizer=g_vae_optimizer, epochs=5, criterion=VAE_Loss)
    #
    # show_images(g_test_examples)
    # g_vae_X_out, _ = g_vae(g_test_examples)
    # show_images(g_vae_X_out)
    #
    # g_vae_optimizer.lr = 0.00001
    # train_and_plot(model=g_vae, optimizer=g_vae_optimizer, epochs=5, criterion=VAE_Loss)
    #
    # show_images(g_test_examples)
    # g_vae_X_out, _ = g_vae(g_test_examples)
    # show_images(g_vae_X_out)
    #
    # g_vae.to(g_device)
    # g_vae_tocanvas = To_Canvas_Format().to(g_device)
    # g_vae_decoder_export = nn.Sequential(g_vae.decoder, g_tocanvas)
    # g_vae_encoder = VAE_to_encoder(g_vae)
    #
    # g_vae_dummy_input, g_vae_dummy_input_y = next(iter(g_test_loader))
    # g_vae_dummy_input = g_vae_dummy_input.to(g_device)
    #
    # g_vae_dummy_latent = g_vae_encoder(g_vae_dummy_input)
    # g_vae_dummy_output = g_vae_decoder_export(g_vae_dummy_latent)
    #
    # export_onnx(model=g_vae_encoder, dummy_input=g_vae_dummy_input, file_name="vae_encoder.onnx")
    # export_onnx(model=g_vae_decoder_export, dummy_input=g_vae_dummy_latent, file_name="vae_decoder.onnx")
