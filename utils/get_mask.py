import torch
import numpy as np

from data import transforms


def get_mask(resolution, return_mask=False, R=4, p_m=False):
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
