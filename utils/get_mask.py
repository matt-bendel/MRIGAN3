import torch
import numpy as np

from data import transforms


def get_mask(resolution, return_mask=False, R=4, p_m=False):
    # a = np.array(
    #     [0, 10, 19, 28, 37, 46, 54, 61, 69, 76, 83, 89, 95, 101, 107, 112, 118, 122, 127, 132, 136, 140, 144, 148,
    #      151, 155, 158, 161, 164,
    #      167, 170, 173, 176, 178, 181, 183, 186, 188, 191, 193, 196, 198, 201, 203, 206, 208, 211, 214, 217, 220,
    #      223, 226, 229, 233, 236,
    #      240, 244, 248, 252, 257, 262, 266, 272, 277, 283, 289, 295, 301, 308, 315, 323, 330, 338, 347, 356, 365,
    #      374])
    # m = np.zeros((384, 384))
    # m[:, a] = True
    # m[:, 176:208] = True
    #
    # samp = m
    # numcoil = 8
    # mask = transforms.to_tensor(np.tile(samp, (numcoil, 1, 1)).astype(np.float32))
    # mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)
    #
    # return mask if return_mask else np.where(m == 1)
    a = np.array(
        [1, 10, 18, 25, 31, 37, 42, 46, 50, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
         76, 80, 84, 88, 93, 99, 105, 112, 120])

    m = np.zeros((128, 128))
    m[:, a] = True
    m[:, 56:72] = True

    samp = m
    numcoil = 8
    mask = transforms.to_tensor(np.tile(samp, (numcoil, 1, 1)).astype(np.float32))
    mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)

    return mask if return_mask else np.where(m == 1)
