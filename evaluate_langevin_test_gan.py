# import required module
import os
import time
import pathlib
import random
import torch
import h5py
import numpy as np
import sigpy as sp
import sigpy.mri as mr
from data.mri_data import SelectiveSliceData_Val
from data_loaders.prepare_data import DataTransform
from evaluation_scripts import compute_cfid
from evaluation_scripts import compute_fid
from utils.fftc import fft2c_new, ifft2c_new
from utils.math import complex_abs, tensor_to_complex_np
from torch.utils.data import DataLoader
from data_loaders.prepare_data import create_data_loaders

from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils.parse_args import create_arg_parser
from wrappers.our_gen_wrapper import load_best_gan


def psnr(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    psnr_val = peak_signal_noise_ratio(gt, pred, data_range=maxval)

    return psnr_val


def snr(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute the Signal to Noise Ratio metric (SNR)"""
    noise_mse = np.mean((gt - pred) ** 2)
    snr = 10 * np.log10(np.mean(gt ** 2) / noise_mse)

    return snr


def ssim(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    # if not gt.ndim == 3:
    #   raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = structural_similarity(
        gt, pred, data_range=maxval
    )

    return ssim

def main(args):
    ref_directory = '/storage/fastMRI_brain/data/small_T2_test'
    # iterate over files in
    # that directory

    # vals = [1, 2, 4, 8, 16, 32]
    vals = [32]

    G = load_best_gan(args)
    G.update_gen_status(val=True)

    train_loader, dev_loader = create_data_loaders(args, big_test=True)
    # compute_cfid.get_cfid(args, G, langevin=True)

    data = SelectiveSliceData_Val(
        root=args.data_path / 'small_T2_test',
        transform=DataTransform(args, test=True),
        challenge='multicoil',
        sample_rate=1,
        use_top_slices=True,
        number_of_top_slices=6,
        restrict_size=False,
        big_test=True
    )

    loader = DataLoader(
        dataset=data,
        batch_size=4,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    print("SMALL CFID")
    # compute_cfid.get_cfid(args, G, langevin=True, loader=loader, ref_loader=None, num_samps=32)

    print("MEDIUM CFID")
    # compute_cfid.get_cfid(args, G, langevin=True, loader=dev_loader, ref_loader=None, num_samps=1)

    print("LARGE CFID")
    # compute_cfid.get_cfid(args, G, langevin=True, loader=dev_loader, ref_loader=train_loader, num_samps=1)

    # compute_fid.get_fid(args, G, train_loader, dev_loader)

    for num in vals:
        num_code = num

        print(f"{num_code} CODE VECTORS")
        psnr_vals = []
        ssim_vals = []
        snr_vals = []
        apsd_vals = []
        times = []


        for i, data in enumerate(loader):
            with torch.no_grad():
                y, x, y_true, mean, std, mask, inds = data
                y = y.to(args.device)
                x = x.to(args.device)
                y_true = y_true.to(args.device)
                mask = mask.cuda()

                gens = torch.zeros(size=(y.size(0), num_code, args.in_chans, args.im_size, args.im_size),
                                   device=args.device)

                for z in range(num_code):
                    start = time.time()
                    gens[:, z, :, :, :] = G(y, y_true, mask=mask)
                    elapsed = time.time() - start
                    times.append(elapsed)

                avg = torch.mean(gens, dim=1)

                temp_gens = torch.zeros(gens.shape, dtype=gens.dtype)
                for z in range(num_code):
                    temp_gens[:, z, :, :, :] = gens[:, z, :, :, :] * std[:, None, None, None].to(args.device) + mean[:,
                                                                                                                None,
                                                                                                                None,
                                                                                                                None].to(
                        args.device)

                apsd_vals.append(torch.mean(torch.std(temp_gens, dim=1), dim=(0, 1, 2, 3)).cpu().numpy())

                new_gens = torch.zeros(y.size(0), num_code, 8, args.im_size, args.im_size, 2)
                new_gens[:, :, :, :, :, 0] = temp_gens[:, :, 0:8, :, :]
                new_gens[:, :, :, :, :, 1] = temp_gens[:, :, 8:16, :, :]

                avg_gen = torch.zeros(size=(y.size(0), 8, args.im_size, args.im_size, 2), device=args.device)
                avg_gen[:, :, :, :, 0] = avg[:, 0:8, :, :]
                avg_gen[:, :, :, :, 1] = avg[:, 8:16, :, :]

                gt = torch.zeros(size=(y.size(0), 8, args.im_size, args.im_size, 2), device=args.device)
                gt[:, :, :, :, 0] = x[:, 0:8, :, :]
                gt[:, :, :, :, 1] = x[:, 8:16, :, :]

                for j in range(y.size(0)):
                    new_y_true = fft2c_new(ifft2c_new(y_true[j]) * std[j] + mean[j])
                    maps = mr.app.EspiritCalib(tensor_to_complex_np(new_y_true.cpu()), calib_width=32,
                                               device=sp.Device(0), show_pbar=False, crop=0.70,
                                               kernel_width=6).run().get()
                    S = sp.linop.Multiply((args.im_size, args.im_size), maps)
                    gt_ksp, avg_ksp = tensor_to_complex_np((gt[j] * std[j] + mean[j]).cpu()), tensor_to_complex_np(
                        (avg_gen[j] * std[j] + mean[j]).cpu())

                    avg_gen_np = torch.tensor(S.H * avg_ksp).abs().numpy()

                    gt_np = torch.tensor(S.H * gt_ksp).abs().numpy()

                    ssim_vals.append(ssim(gt_np, avg_gen_np))
                    psnr_vals.append(psnr(gt_np, avg_gen_np))
                    snr_vals.append(snr(gt_np, avg_gen_np))

        print('AVERAGE')
        print(f'APSD: {np.mean(apsd_vals)} \pm {np.std(apsd_vals) / np.sqrt(len(apsd_vals))}')
        print(f'PSNR: {np.mean(psnr_vals)} \pm {np.std(psnr_vals) / np.sqrt(len(psnr_vals))}')
        print(f'SNR: {np.mean(snr_vals)} \pm {np.std(snr_vals) / np.sqrt(len(snr_vals))}')
        print(f'SSIM: {np.mean(ssim_vals)} \pm {np.std(ssim_vals) / np.sqrt(len(ssim_vals))}')
        print(f'TIME: {np.mean(times)}')
        print("\n")
        # print('MEDIAN')
        # print('APSD: ', np.median(apsd_vals))
        # print('PSNR: ', np.median(psnr_vals))
        # print('SNR: ', np.median(snr_vals))
        # print('SSIM: ', np.median(ssim_vals))
        print("\n")


if __name__ == '__main__':
    cuda = True if torch.cuda.is_available() else False

    args = create_arg_parser().parse_args()
    # restrict visible cuda devices
    if args.data_parallel or (args.device >= 0):
        if not args.data_parallel:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.in_chans = 16
    args.out_chans = 16

    # args.checkpoint_dir = "/home/bendel.8/Git_Repos/MRIGAN3/trained_models/cvpr_ours"
    # main(args)

    # args.checkpoint_dir = "/home/bendel.8/Git_Repos/MRIGAN3/trained_models/cvpr_ohayon"
    # main(args)

    args.checkpoint_dir = "/home/bendel.8/Git_Repos/MRIGAN3/trained_models/cvpr_adler"
    main(args)

    # print('MEDIAN')
    # print('APSD: ', np.median(apsd_vals))
    # print('PSNR: ', np.median(psnr_vals))
    # print('SNR: ', np.median(snr_vals))
    # print('SSIM: ', np.median(ssim_vals))