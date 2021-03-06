import sys, os, glob
import h5py

import numpy as np
import sigpy as sp
import cupy as cp
import sigpy.mri as mr
import sigpy.plot as pl
import matplotlib.pyplot as plt

from data.transforms import to_tensor
from utils.espirit import ifft, fft


def crop_and_compress(x):
    w_from = (x.shape[0] - 384) // 2  # crop images into 384x384
    h_from = (x.shape[1] - 384) // 2
    w_to = w_from + 384
    h_to = h_from + 384
    cropped_x = x[w_from:w_to, h_from:h_to, :]
    if cropped_x.shape[-1] > 8:
        x_tocompression = cropped_x.reshape(384 ** 2, cropped_x.shape[-1])
        U, S, Vh = np.linalg.svd(x_tocompression, full_matrices=False)
        coil_compressed_x = np.matmul(x_tocompression, Vh.conj().T)
        coil_compressed_x = coil_compressed_x[:, 0:8].reshape(384, 384, 8)
    else:
        coil_compressed_x = cropped_x

    return coil_compressed_x


def apply_mask(y, R):
    if R == 8:
        a = np.array(
            [1, 24, 45, 64, 81, 97, 111, 123, 134, 144, 153, 161, 168, 175, 181, 183, 184, 185, 186, 187, 188, 189, 190,
             191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 205, 211, 218, 225, 233, 242, 252, 263, 275, 289,
             305, 322, 341, 362])
        m = np.zeros((384, 384))
        m[:, a] = True
        m[:, 183:201] = True
    else:
        a = np.array(
            [0, 10, 19, 28, 37, 46, 54, 61, 69, 76, 83, 89, 95, 101, 107, 112, 118, 122, 127, 132, 136, 140, 144, 148,
             151, 155, 158, 161, 164,
             167, 170, 173, 176, 178, 181, 183, 186, 188, 191, 193, 196, 198, 201, 203, 206, 208, 211, 214, 217, 220,
             223, 226, 229, 233, 236,
             240, 244, 248, 252, 257, 262, 266, 272, 277, 283, 289, 295, 301, 308, 315, 323, 330, 338, 347, 356, 365,
             374])
        m = np.zeros((384, 384))
        m[:, a] = True
        m[:, 176:208] = True

    samp = m
    numcoil = 8
    mask = np.tile(samp, (numcoil, 1, 1)).astype(np.float32)

    return y * mask


def main(R, data):
    in_dir = f'/storage/fastMRI_brain/data/multicoil_{data}'
    out_dir = f'/storage/fatMRI_brain_ls/{data}_R={R}'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for file in sorted(glob.glob(in_dir + '/*.h5')):
        basename = os.path.basename(file)
        out_name = os.path.join(out_dir, basename)

        if os.path.exists(out_name):
            print('CONT')
            continue

        with h5py.File(file, 'r') as data:
            if data.attrs['acquisition'] != 'AXT2' or data['kspace'].shape[1] < 8 or data['kspace'].shape[-1] < 384:
                print('NEXT')
                continue

            print(basename)

            kspace = data['kspace']
            recons = np.zeros((kspace.shape[0], 8, 384, 384), dtype=kspace.dtype)
            gt = np.zeros((kspace.shape[0], 8, 384, 384), dtype=kspace.dtype)
            s_maps = np.zeros((kspace.shape[0], 8, 384, 384), dtype=kspace.dtype)

            for i in range(kspace.shape[0]):
                x = ifft(kspace[i, :, :, :], (1, 2))  # (slices, num_coils, H, W)
                coil_compressed_x = crop_and_compress(x.transpose(1, 2, 0)).transpose(2, 0, 1)
                y = apply_mask(fft(coil_compressed_x, (1, 2)), R)
                # zfr = ifft(y, (1, 2))
                # pl.ImagePlot(zfr, z=0, title='Multicoil ZFR')
                # plt.savefig('temp0.png')

                s_map = mr.app.EspiritCalib(y, calib_width=32, show_pbar=True, crop=0.7, kernel_width=5,
                                            device=sp.Device(1)).run()

                x_ls = mr.app.L1WaveletRecon(y, s_map, lamda=1e-10, show_pbar=True, device=sp.Device(1)).run()

                sense_op = sp.linop.Multiply((384, 384), s_map)
                # pl.ImagePlot(sense_op.H * coil_compressed_x, title='LS Recon', save_basename='temp')
                # plt.savefig('temp1.png')
                # F = sp.linop.FFT(y.shape, axes=(-1, -2))
                # multi_zfr = sense_op.H * F.H * y
                # pl.ImagePlot(multi_zfr, title='ZFR')
                # plt.savefig('temp00.png')
                x_ls_multicoil = sense_op * x_ls

                # pl.ImagePlot(x_ls_multicoil, z=0, title='Multicoil LS Recon')
                # plt.savefig('temp2.png')
                recons[i, :, :, :] = cp.asnumpy(x_ls_multicoil)
                s_maps[i, :, :, :] = cp.asnumpy(s_map)
                gt[i, :, :, :] = coil_compressed_x

            h5 = h5py.File(out_name, 'w')
            h5.create_dataset('gt', data=gt)
            h5.create_dataset('ls_recons', data=recons)
            h5.create_dataset('sense_maps_operator', data=s_maps)
            h5.close()


if __name__ == '__main__':
    main(4, 'train')
    main(4, 'val')
