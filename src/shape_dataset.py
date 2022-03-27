import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from scipy import ndimage
from tqdm import tqdm

HEART = torch.tensor([
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
]).float()

ARROW = torch.tensor([
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
]).float()

SIMPLE_ARROW = torch.tensor([
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
]).float()

SQUIGGLE = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    [1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

]).float()

BOXEY_THING = torch.tensor([
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

]).float()


def get_shape_image_4(gen_vars: torch.tensor) -> torch.Tensor:
    # gen_vars = [x, y, scale, rotation]
    assert gen_vars.shape == (4,), gen_vars.shape
    coords = (5 + (1.5 * gen_vars[0:2])).int()
    unders = coords <= 0
    coords[unders] = 0
    overs = coords >= 10
    coords[overs] = 10
    x1, y1 = coords

    scale = 1 + (float(gen_vars[2] * 0.1))
    if scale > 1.30:
        scale = 1.30
    if scale < 0.7:
        scale = 0.7

    rot = gen_vars[3] * 90.0

    im1 = ndimage.zoom(BOXEY_THING, zoom=scale)
    im1[im1 < 0] = 0.0
    im1 /= im1.max()
    im1 = torch.tensor(ndimage.rotate(im1, angle=rot))
    im1[im1 < 0] = 0.0
    im1 /= im1.max()

    h1 = im1.shape[0]
    w1 = im1.shape[1]
    assert h1 <= 18, h1
    assert w1 <= 18, w1

    blank = torch.zeros((1, 28, 28))
    blank[0, x1:x1 + h1, y1:y1 + w1] += im1

    return blank


class ShapesDataset(Dataset):
    def __init__(self, N=int(1e4)):
        self.N = N
        self.num_generatives = 4
        self.gen_vars = torch.normal(0.0, 1, size=(N, self.num_generatives))
        self.im_y_s = [self.get_im_(index) for index in range(N)]
        self.ims = torch.stack([im for im, _ in self.im_y_s], dim=0)
        mean = self.ims.mean()
        std = self.ims.std()
        self.ims = (self.ims - mean) / std
        self.ys = [y for _, y in self.im_y_s]

    def get_im_(self, index):
        y = self.gen_vars[index, :]
        return get_shape_image_4(y), y

    def __getitem__(self, index) -> T_co:
        return self.ims[index], self.ys[index]

    def __len__(self) -> T_co:
        return self.N


def get_im_pairs_tensor(B: int, L: int):
    # Notation taken from beta-VAE paper
    # B = number of samples (final Z_b vectors produced)
    # L = number of pairs used to crete a single sample
    # 2 = number in a pair
    # 1 = number of colour channels
    # 28,28 = h * w
    im_tensor = torch.empty(B, L, 2, 1, 28, 28)
    ys = torch.randint(0, 4, size=(B,))

    # Randomly choose the generative variables
    gen_vars = torch.normal(0.0, 1, size=(B, L, 2, 4))
    for b in range(B):
        gen_vars[b, :, 0, ys[b]] = gen_vars[b, :, 1, ys[b]]

    # Generate the images
    for b in range(B):
        for l in range(L):
            im1 = get_shape_image_4(gen_vars[b, l, 0])
            im_tensor[b, l, 0, :, :, :] = im1
            im2 = get_shape_image_4(gen_vars[b, l, 1])
            im_tensor[b, l, 1, :, :, :] = im2

    mean = im_tensor.mean()
    std = im_tensor.std()
    im_tensor = (im_tensor - mean) / std
    return im_tensor, ys


class ZbDatset(Dataset):
    def __init__(self, encoder, im_tensor: torch.Tensor, ys: torch.tensor):
        # Notation taken from beta-VAE paper
        # B = number of samples (final Z_b vectors produced)
        # L = number of pairs used to crete a single sample
        # 2 = number in a pair
        # 1 = number of colour channels
        # 28,28 = h * w
        self.B, self.L = im_tensor.shape[0], im_tensor.shape[1]
        assert im_tensor.shape == (self.B, self.L, 2, 1, 28, 28), (im_tensor.shape, (self.B, self.L, 2, 1, 28, 28))
        assert ys.shape == (self.B,), (ys.shape, self.B)
        self.num_generatives = 4

        self.im_tensor = im_tensor

        self.latent_size = encoder.latent_size
        self.encoder = encoder

        self.zs = torch.empty(self.B, self.L, 2, self.latent_size)

        self.ys = ys

        for b in range(self.B):
            im_batch = self.im_tensor[b, :]
            with torch.no_grad():
                self.zs[b, :, 0] = encoder(im_batch[:, 0]).detach()
                self.zs[b, :, 1] = encoder(im_batch[:, 1]).detach()

        z_diffs = (self.zs[:, :, 0] - self.zs[:, :, 1]).abs()
        self.z_bs = z_diffs.mean(dim=1)
        assert self.z_bs.shape == (self.B, self.latent_size)

    def __getitem__(self, index) -> T_co:
        return self.z_bs[index], self.ys[index]

    def __len__(self) -> T_co:
        return self.B


if __name__ == "__main__":
    from src.disentanglement_experiments.quantify_disentanglement import get_encoder_and_info_from_neptune

    g_B = 1000
    g_L = 25
    g_ims, g_ys = get_im_pairs_tensor(g_B, g_L)

    g_encoder, _ = get_encoder_and_info_from_neptune("BVAE-423")

    g_ds = ZbDatset(g_encoder, g_ims, g_ys)
    g_vecs, g_ys = g_ds[:]
    for i in range(g_B):
        print(g_vecs[i])
