import math

from matplotlib import pyplot as plt


def show_images(X):
    fig = plt.figure()
    X = X.detach().to('cpu')
    n = X.shape[0]
    nrows = 3
    ncols = math.ceil(n / nrows)

    for i in range(n):
        plt.subplot(nrows, ncols, i + 1)
        plt.tight_layout()
        plt.imshow(X[i][0], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
    plt.show()


def visualize_latent_space(X, encoder):
    X = X.to(next(encoder.parameters()).device)
    Z = encoder(X)
    Z = Z.to('cpu').detach()
    plt.scatter(Z[:, 0], Z[:, 1])


# class To_Canvas_Format(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         x = x[0, :]  # only one image, not whole batch
#         x = torch.mul(x, 255).type(torch.uint8).repeat(3, 1, 1)  # from grayscale to rgb
#         x = F.pad(x, (0, 0, 0, 0, 0, 1), value=255)  # from rgb to rgba
#
#         x = torch.permute(x, (1, 2, 0))  # set to pixel.x, pixel.y, rgba
#         return x

#
# def export_onnx(model, dummy_input, file_name):
#     torch.onnx.export(model,  # model being run
#                       dummy_input,  # model input (or a tuple for multiple inputs)
#                       file_name,  # where to save the model (can be a file or file-like object)
#                       export_params=True,  # store the trained parameter weights inside the model file
#                       opset_version=14,  # the ONNX version to export the model to
#                       do_constant_folding=True,  # whether to execute constant folding for optimization
#                       input_names=['input'],  # the model's input names
#                       output_names=['output'],  # the model's output names
#                       dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
#                                     'output': {0: 'batch_size'}})
#
#
# def export_data_to_json(dataloader, n, name):
#     export_list_X = []
#     export_list_Y = []
#     for X, Y in dataloader:
#         taking = min(len(X), n - len(export_list_X))
#         if taking <= 0:
#             break
#         export_list_X.extend(X.tolist())
#         export_list_Y.extend(Y.tolist())
#     json.dump(export_list_X, codecs.open(name + "_X.json", 'w', encoding='utf-8'),
#               separators=(',', ':'),
#               indent=4)
#     json.dump(export_list_Y, codecs.open(name + "_Y.json", 'w', encoding='utf-8'),
#               separators=(',', ':'),
#               indent=4)
