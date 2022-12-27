import torch
import numpy as np

from data import transforms


def get_mask(resolution, R, mask_num):
    total_lines = resolution // R - 10
    m = np.zeros((resolution, resolution))
    midway = resolution // 2
    s = midway - 10 // 2
    e = s + 10
    m[:, s:e] = True
    a = np.random.choice(resolution - 10, total_lines, replace=False)
    a = np.where(a < s, a, a + 10)
    m[:, a] = True

    with open(f'mask_study/mask_{mask_num}.npy', 'wb') as f:
        np.save(f, m)

for i in range(100):
    get_mask(128, 4, i)
