# import required module
import os
import pathlib
import torch
import numpy as np
from utils.math import complex_abs
from utils.parse_args import create_arg_parser
from evaluation_scripts.fid.embeddings import VGG16Embedding

from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from evaluation_scripts.fid.fid_metric_langevin_avg import FIDMetric
from evaluation_scripts.cfid.cfid_metric_langevin import CFIDMetric
from data_loaders.prepare_data import create_data_loaders
import lpips
import pickle
from DISTS_pytorch import DISTS

# M: 2.61
# C: 4.73
# CFID: 7.34

# FID: 20.52

# APSD: 7.6e-6
# PSNR: 33.90 \pm 0.45
# SSIM: 0.8589 \pm 0.0083

def get_cfid(args, G, ref_loader, cond_loader):
    print("GETTING INCEPTION EMBEDDING")
    vgg_embedding = VGG16Embedding(parallel=True)

    print("GETTING DATA LOADERS")

    cfid_metric = CFIDMetric(gan=None,
                           ref_loader=ref_loader,
                           loader=None,
                           image_embedding=vgg_embedding,
                           condition_embedding=vgg_embedding,
                           cuda=True,
                           args=args)

    cfid = cfid_metric.get_cfid_torch_pinv()
    print(f"CFID: {cfid}")

def get_fid(args, G, ref_loader, cond_loader, num_samps=32):
    print("GETTING INCEPTION EMBEDDING")
    vgg_embedding = VGG16Embedding(parallel=True)

    print("GETTING DATA LOADERS")

    fid_metric = FIDMetric(gan=None,
                           ref_loader=ref_loader,
                           loader=None,
                           image_embedding=vgg_embedding,
                           condition_embedding=vgg_embedding,
                           cuda=True,
                           args=args,
                           num_samps=num_samps)

    fid_metric.get_fid()

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


def unnormalize(gen_img, estimated_mvue):
    '''
        Estimate mvue from coils and normalize with 99% percentile.
    '''
    scaling = torch.quantile(estimated_mvue.abs(), 0.99)
    return gen_img / scaling

def rgb(im, unit_norm=False):
    embed_ims = torch.zeros(size=(3, 384, 384))
    tens_im = torch.tensor(im)

    if unit_norm:
        tens_im = (tens_im - torch.min(tens_im)) / (torch.max(tens_im) - torch.min(tens_im))
    else:
        tens_im = 2 * (tens_im - torch.min(tens_im)) / (torch.max(tens_im) - torch.min(tens_im)) - 1

    embed_ims[0, :, :] = tens_im
    embed_ims[1, :, :] = tens_im
    embed_ims[2, :, :] = tens_im

    return embed_ims.unsqueeze(0)

R = 8

# assign directory
ref_directory = '/storage/fastMRI_brain/data/small_T2_test'
recon_directory = f'/storage/fastMRI_brain/Langevin_Recons_R={R}/'
# iterate over files in
# that directory

args = create_arg_parser().parse_args()

train_loader, _ = create_data_loaders(args, big_test=False)
get_cfid(args, None, train_loader, None)
# n_samps = [1, 2, 4, 8, 16, 32]
# for n in n_samps:
#     print(f"{n} SAMPLES")
get_fid(args, None, train_loader, None, num_samps=32)
# exit()
vals = [1, 2, 4, 8, 16, 32]
lpips_met = lpips.LPIPS(net='alex')
dists_met = DISTS()

exceptions = False
count = 0
n_psnrs = []
n_ssims = []
n_distss = []
n_lpipss = []

for k in vals:
    print(f"{k} CODE VECTORS")
    psnr_vals = []
    ssim_vals = []
    snr_vals = []
    apsd_vals = []
    lpips_vals = []
    dists_vals = []

    for filename in os.listdir(ref_directory):
        for i in range(6):
            recons = np.zeros((k, 384, 384))
            recon_object = None

            maps = None
            with open(f'/storage/fastMRI_brain/sense_maps/test_full_res/{filename}_{i}.pkl', 'rb') as inp:
                maps = pickle.load(inp)

            test_mask = maps[0, :, :] > 1e-3

            for j in range(k):
                try:
                    new_filename = recon_directory + filename + f'|langevin|slide_idx_{i}_R={R}_sample={j}_outputs.pt'
                    recon_object = torch.load(new_filename)
                    count += 1
                except:
                    print(filename)
                    exceptions = True
                    continue
                # temp_recon = unnormalize(recon_object['mvue'], recon_object['zfr'])

                recons[j] = complex_abs(recon_object['mvue'][0].permute(1, 2, 0)).cpu().numpy()

            if exceptions:
                print("EXCEPTION")
                exceptions = False
                continue

            mean = np.mean(recons, axis=0) * test_mask
            gt = recon_object['gt'][0][0].abs().cpu().numpy() * test_mask
            apsd = np.mean(np.std(recons, axis=0), axis=(0, 1))

            apsd_vals.append(apsd)
            psnr_vals.append(psnr(gt, mean))
            snr_vals.append(snr(gt, mean))
            ssim_vals.append(ssim(gt, mean))
            with torch.no_grad():
                lpips_vals.append(lpips_met(rgb(gt), rgb(mean)).numpy())
                dists_vals.append(dists_met(rgb(gt, unit_norm=True), rgb(mean, unit_norm=True)))

    # print('AVERAGE')
    n_psnrs.append(np.mean(psnr_vals))
    n_ssims.append(np.mean(ssim_vals))
    n_lpipss.append(np.mean(lpips_vals))
    n_distss.append(np.mean(dists_vals))

psnr_str = ''
ssim_str = ''
lpips_str = ''
dists_str = ''

for i in range(len(n_psnrs)):
    psnr_str = f'{psnr_str} {n_psnrs[i]:.2f} &'
    ssim_str = f'{ssim_str} {n_ssims[i]:.4f} &'
    lpips_str = f'{lpips_str} {n_lpipss[i]:.4f} &'
    dists_str = f'{dists_str} {n_distss[i]:.4f} &'

print("PSNR and SSIM:")
print(f'{psnr_str} {ssim_str}')
print("LPIPS and DISTS")
print(f'{lpips_str} {dists_str}')
    # # print(f'APSD: {np.mean(apsd_vals)} \pm {np.std(apsd_vals) / np.sqrt(len(apsd_vals))}')
    # print(f'PSNR: {np.mean(psnr_vals)} \pm {np.std(psnr_vals) / np.sqrt(len(psnr_vals))}')
    # # print(f'SNR: {np.mean(snr_vals)} \pm {np.std(snr_vals) / np.sqrt(len(snr_vals))}')
    # print(f'SSIM: {np.mean(ssim_vals)} \pm {np.std(ssim_vals) / np.sqrt(len(ssim_vals))}')
    # # print(f'LPIPS: {np.mean(lpips_vals)} \pm {np.std(lpips_vals) / np.sqrt(len(lpips_vals))}')
    # print(f'DISTS: {np.mean(dists_vals)} \pm {np.std(dists_vals) / np.sqrt(len(dists_vals))}')
    #
    # print("\n")

    # print('MEDIAN')
    # print('APSD: ', np.median(apsd_vals))
    # print('PSNR: ', np.median(psnr_vals))
    # print('SNR: ', np.median(snr_vals))
    # print('SSIM: ', np.median(ssim_vals))