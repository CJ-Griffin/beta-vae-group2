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


def get_shape_image(params: torch.tensor):
    if params.shape == (8,):
        return get_shape_image_8(params)
    elif params.shape == (4,):
        return get_shape_image_4(params)


def get_shape_image_8(params: torch.tensor):
    assert params.shape == (8,), params.shape
    coords = (5 + (3 * params[0:4])).int()
    unders = coords <= 0
    coords[unders] = 0
    overs = coords >= 10
    coords[overs] = 10
    x1, x2, y1, y2 = coords

    colours = (params[6:8] / 4.0) + 0.25
    unders2 = colours <= 0.2
    colours[unders2] = 0.2
    overs2 = colours >= 0.5
    colours[overs2] = 0.5

    b1, b2 = colours

    rot1, rot2 = params[4:6] * 45

    # im1 = torch.tensor(ndimage.rotate(HEART, angle=rot1)) * b1
    im1 = torch.tensor(ndimage.zoom(ndimage.rotate(HEART, angle=rot1), 1.3)) * b1
    im2 = torch.tensor(ndimage.zoom(ndimage.rotate(ARROW, angle=rot2), 1.3)) * b2
    h1 = im1.shape[0]
    w1 = im1.shape[1]
    h2 = im2.shape[0]
    w2 = im2.shape[1]
    blank = torch.zeros((1, 28, 28))
    blank[0, x1:x1 + h1, y1:y1 + w1] += im1
    # blank[0, x2:x2 + h2, y2:y2 + w2] += im2
    return blank


def get_shape_image_4(params: torch.tensor):
    assert params.shape == (4,), params.shape
    coords = (5 + (1.5 * params[0:2])).int()
    unders = coords <= 0
    coords[unders] = 0
    overs = coords >= 10
    coords[overs] = 10
    x1, y1 = coords

    scale = 1 + (float(params[2] * 0.1))
    if scale > 1.30:
        scale = 1.30
    if scale < 0.7:
        scale = 0.7

    rot = params[3] * 90.0

    # colours = (params[3] / 4.0) + 0.25
    # unders2 = colours <= 0.1
    # colours[unders2] = 0.1
    # overs2 = colours >= 0.9
    # colours[overs2] = 0.9

    # b1 = 1.0 # colours.item()
    im1 = ndimage.zoom(BOXEY_THING, zoom=scale)
    im1[im1 < 0] = 0.0
    im1 /= im1.max()
    im1 = torch.tensor(ndimage.rotate(im1, angle=rot))
    im1[im1 < 0] = 0.0
    im1 /= im1.max()
    # min = im1.min()
    # if min < 0:
    #     im1 -= min
    # new_max = im1.max()
    # im1 /= new_max
    # im1 = torch.tensor(ndimage.rotate(BOXEY_THING, angle=rot1))
    # print(im1.min(), 888)
    # print(im1.max(), 999)
    # im1 = im1 * b1
    # im2 = torch.tensor(ndimage.zoom(ndimage.rotate(ARROW, angle=rot2), 1.3)) * b2
    h1 = im1.shape[0]
    w1 = im1.shape[1]
    assert h1 <= 18, h1
    assert w1 <= 18, w1
    # h2 = im2.shape[0]
    # w2 = im2.shape[1]
    # print(h1, h2)
    blank = torch.zeros((1, 28, 28))
    blank[0, x1:x1 + h1, y1:y1 + w1] += im1
    # blank[0, x2:x2 + h2, y2:y2 + w2] += im2
    # print(blank.min())
    # print(blank.max())
    return blank


class ShapesDataset(Dataset):
    def __init__(self, N=int(1e4), num_generatives=4):
        self.N = N
        self.num_generatives = num_generatives
        self.params = torch.normal(0.0, 1, size=(N, num_generatives))
        self.im_y_s = [self.get_im_(index) for index in range(N)]
        self.ims = torch.stack([im for im, _ in self.im_y_s], dim=0)
        mean = self.ims.mean()
        std = self.ims.std()
        self.ims = (self.ims - mean) / std
        self.ys = [y for _, y in self.im_y_s]

    def get_im_(self, index):
        y = self.params[index, :]
        if self.num_generatives == 3:
            y1 = torch.cat([y, torch.tensor([1])])
        else:
            y1 = y
        return get_shape_image(y1), y

    def __getitem__(self, index) -> T_co:
        return self.ims[index], self.ys[index]

    def __len__(self) -> T_co:
        return self.N


class ShapesDatasetDisentanglementPairs(ShapesDataset):
    def __init__(self, y_ind: int, N=int(1e4), num_generatives=4):
        super().__init__(N, num_generatives)
        self.y_ind = y_ind
        self.rand_indeces = np.random.randint(0, self.N, size=self.N)

    def __getitem__(self, index: int):
        if not isinstance(index, int):
            raise NotImplementedError(type(index))
        params = self.params[index, :].clone()
        params2 = self.params[self.rand_indeces[index], :].clone()
        params2[self.y_ind] = params[self.y_ind]
        return get_shape_image(params), get_shape_image(params2)


class ZbDatsetOld(Dataset):
    def __init__(self, encoder, N=int(1e3), L=25):
        self.N = N
        assert N % 4 == 0, N
        B = N // 4
        self.B = B
        self.num_generatives = 4
        self.L = L
        self.params = []
        self.ims = []
        self.latent_size = encoder.latent_size
        self.encoder = encoder
        self.z_bs = torch.zeros(N, 10)
        self.y_inds = torch.zeros(N)
        for y_ind in tqdm(range(4)):
            # Shape is: number of samples, num pairs per sample, 2 (for pairs), 4 for num gen variables
            params = torch.normal(0.0, 1, size=(B, L, 2, 4))
            params[:, :, 0, y_ind] = params[:, :, 1, y_ind]
            print(params[0, 0])
            self.params.append(params)
            im_tensor = torch.zeros(B, L, 2, 1, 28, 28)
            encoding_tensor = torch.zeros(B, L, 2, self.latent_size)
            # Generate the images
            for b in range(B):
                for l in range(L):
                    im1 = get_shape_image_4(params[b, l, 0])
                    im_tensor[b, l, 0, :, :, :] = im1
                    im2 = get_shape_image_4(params[b, l, 1])
                    im_tensor[b, l, 1, :, :, :] = im2
                    # plt.imshow(torch.vstack([im1.view(28,28), im2.view(28,28)]))
                    # plt.show()

            mean = im_tensor.mean()
            std = im_tensor.std()
            # print(im_tensor.max(), im_tensor.min(), 1)
            im_tensor = (im_tensor - mean) / std
            im_tensor = im_tensor
            # print(im_tensor.max(), im_tensor.min(), 2)

            for b in range(B):
                for one_or_two in range(2):
                    batch = im_tensor[b, :, one_or_two]
                    # print(batch.shape)
                    encodings = self.encoder(batch).detach()
                    # print(encodings.shape)
                    encoding_tensor[b, :, one_or_two, :] = encodings
            # for i in range(10):
            #     plt.imshow(im_tensor[0, i, 0].view(28, 28))
            #     plt.title(encoding_tensor[0, i, 0])
            #     plt.show()
            #     plt.imshow(im_tensor[0, i, 1].view(28, 28))
            #     plt.title(encoding_tensor[0, i, 1])
            #     plt.show()
            #     print(encoding_tensor[0, i, 0, :])
            #     print(encoding_tensor[0, i, 1, :])
            #     print()
            # print(encoding_tensor.max(), encoding_tensor.min(), 3)
            z_diffs = (encoding_tensor[:, :, 0, :] - encoding_tensor[:, :, 1, :]).abs()
            # print(z_diffs.max(), z_diffs.min(), z_diffs.mean(), 4)
            z_bs = z_diffs.mean(dim=1)
            # print(z_bs.max(), z_bs.min(), z_bs.mean(), 5)
            self.z_bs[B * y_ind:B * (y_ind + 1), :] = z_bs
            self.y_inds[B * y_ind:B * (y_ind + 1)] = y_ind

    def __getitem__(self, index) -> T_co:
        return self.z_bs[index], self.y_inds[index]

    def __len__(self) -> T_co:
        return self.N


def get_im_pairs_tensor(B: int, L: int):
    im_tensor = torch.empty(B, L, 2, 1, 28, 28)
    ys = torch.randint(0, 4, size=(B,))
    # Shape is: number of samples, num pairs per sample, 2 (for pairs), 4 for num gen variables
    params = torch.normal(0.0, 1, size=(B, L, 2, 4))
    for b in range(B):
        params[b, :, 0, ys[b]] = params[b, :, 1, ys[b]]

    # Generate the images
    for b in range(B):
        for l in range(L):
            im1 = get_shape_image_4(params[b, l, 0])
            im_tensor[b, l, 0, :, :, :] = im1
            im2 = get_shape_image_4(params[b, l, 1])
            # print(params[b, l, 0] - params[b, l, 1])
            im_tensor[b, l, 1, :, :, :] = im2

    mean = im_tensor.mean()
    std = im_tensor.std()
    im_tensor = (im_tensor - mean) / std
    return im_tensor, ys


class ZbDatset(Dataset):
    def __init__(self, encoder, im_tensor: torch.Tensor, ys: torch.tensor):
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
