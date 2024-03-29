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
from evaluation_scripts.fid.fid_metric_langevin import FIDMetric
from evaluation_scripts.cfid.cfid_metric_langevin import CFIDMetric
from data_loaders.prepare_data import create_data_loaders

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

def get_fid(args, G, ref_loader, cond_loader):
    print("GETTING INCEPTION EMBEDDING")
    vgg_embedding = VGG16Embedding(parallel=True)

    print("GETTING DATA LOADERS")

    fid_metric = FIDMetric(gan=None,
                           ref_loader=ref_loader,
                           loader=None,
                           image_embedding=vgg_embedding,
                           condition_embedding=vgg_embedding,
                           cuda=True,
                           args=args)

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


R = 4

# assign directory
ref_directory = '/storage/fastMRI_brain/data/small_T2_test'
recon_directory = f'/storage/fastMRI_brain/Langevin_Recons_R={R}/'
# iterate over files in
# that directory

args = create_arg_parser().parse_args()

train_loader, _ = create_data_loaders(args, big_test=False)
get_cfid(args, None, train_loader, None)
exit()

vals = [32]

exceptions = False
count = 0
for k in vals:
    print(f"{k} CODE VECTORS")
    psnr_vals = []
    ssim_vals = []
    snr_vals = []
    apsd_vals = []
    for filename in os.listdir(ref_directory):
        for i in range(6):
            recons = np.zeros((k, 384, 384))
            recon_object = None
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
            mean = np.mean(recons, axis=0)
            gt = recon_object['gt'][0][0].abs().cpu().numpy()
            apsd = np.mean(np.std(recons, axis=0), axis=(0, 1))

            apsd_vals.append(apsd)
            psnr_vals.append(psnr(gt, mean))
            snr_vals.append(snr(gt, mean))
            ssim_vals.append(ssim(gt, mean))

    print('AVERAGE')
    print(f'APSD: {np.mean(apsd_vals)} \pm {np.std(apsd_vals) / np.sqrt(len(apsd_vals))}')
    print(f'PSNR: {np.mean(psnr_vals)} \pm {np.std(psnr_vals) / np.sqrt(len(psnr_vals))}')
    print(f'SNR: {np.mean(snr_vals)} \pm {np.std(snr_vals) / np.sqrt(len(snr_vals))}')
    print(f'SSIM: {np.mean(ssim_vals)} \pm {np.std(ssim_vals) / np.sqrt(len(ssim_vals))}')
    print("\n")

print(count)
    # print('MEDIAN')
    # print('APSD: ', np.median(apsd_vals))
    # print('PSNR: ', np.median(psnr_vals))
    # print('SNR: ', np.median(snr_vals))
    # print('SSIM: ', np.median(ssim_vals))