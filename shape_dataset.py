import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from scipy import ndimage

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
    coords = (5 + (3 * params[0:2])).int()
    unders = coords <= 0
    coords[unders] = 0
    overs = coords >= 10
    coords[overs] = 10
    x1, y1 = coords

    rot1 = params[2] * 45

    colours = (params[3] / 4.0) + 0.25
    unders2 = colours <= 0.1
    colours[unders2] = 0.1
    overs2 = colours >= 1.9
    colours[overs2] = 1.9

    b1 = colours.item()

    # im1 = torch.tensor(ndimage.rotate(HEART, angle=rot1)) * b1
    im1 = torch.tensor(ndimage.zoom(ndimage.rotate(BOXEY_THING, angle=rot1), 1.3)) * b1
    # im2 = torch.tensor(ndimage.zoom(ndimage.rotate(ARROW, angle=rot2), 1.3)) * b2
    h1 = im1.shape[0]
    w1 = im1.shape[1]
    # h2 = im2.shape[0]
    # w2 = im2.shape[1]
    # print(h1, h2)
    blank = torch.zeros((1, 28, 28))
    blank[0, x1:x1 + h1, y1:y1 + w1] += im1
    # blank[0, x2:x2 + h2, y2:y2 + w2] += im2
    return blank


class ShapesDataset(Dataset):
    def __init__(self, N=int(1e5), num_generatives=3):
        self.N = N
        self.num_generatives = num_generatives
        self.params = torch.normal(0.0, 0.001, size=(N, num_generatives))
        self.im_y_s = [self.get_im_(self.params[index]) for index in range(N)]

    def get_im_(self, index):
        y = self.params[index]
        if self.num_generatives == 3:
            y1 = torch.cat([y, torch.tensor([1])])
        else:
            y1 = y
        return get_shape_image(y1), y

    def __getitem__(self, index) -> T_co:
        return self.im_y_s[index]

    def __len__(self) -> T_co:
        return self.N


if __name__ == "__main__":
    ds = ShapesDataset(num_generatives=3)
    # plt.imshow(SQUIGGLE, vmin=0.0, vmax=1.0)
    # plt.show()
    for i in range(10):
        im, y = ds[i]
        # x,y,b = np.random.uniform(0.0, 1.0, size=3)
        # rot = np.random.choice([0,1,2,3])
        plt.imshow(im[0], vmin=0.0, vmax=1.0)
        print(y)
        plt.show()

    print(HEART.shape)
