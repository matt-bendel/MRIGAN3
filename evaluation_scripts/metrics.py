import torch
import os
import numpy as np
import sigpy as sp
from evaluation_scripts import compute_cfid
import matplotlib.pyplot as plt
import imageio as iio
import sigpy.mri as mr

from typing import Optional
from wrappers.our_gen_wrapper import load_best_gan
from data_loaders.prepare_data import create_data_loaders
from data_loaders.prepare_data_ls import create_data_loaders_ls
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.decomposition import PCA
from utils.math import tensor_to_complex_np
from utils.fftc import ifft2c_new, fft2c_new
from data import transforms
from utils.math import complex_abs
from mail import send_mail

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


def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute the Signal to Noise Ratio metric (SNR)"""
    noise_mse = np.mean((gt - pred) ** 2)

    return noise_mse


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


def generate_image(fig, target, image, method, image_ind, rows, cols, kspace=False, disc_num=False):
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(rows, cols, image_ind)

    if method != 'GT' and method != 'Std. Dev':
        psnr_val = psnr(target, image)
        snr_val = snr(target, image)
        ssim_val = ssim(target, image)
        if not kspace:
            pred = disc_num
            ax.text(1, 0.8, f'PSNR: {psnr_val:.2f}\nSNR: {snr_val:.2f}\nSSIM: {ssim_val:.4f}', transform=ax.transAxes,
                    horizontalalignment='right', verticalalignment='center', fontsize='xx-small', color='yellow')

    if method == 'Std. Dev':
        im = ax.imshow(image, cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_title(method, size=10)
        if kspace:
            image = image ** 0.4
            target = target ** 0.4
        im = ax.imshow(np.abs(image), cmap='gray', vmin=0, vmax=np.max(target))
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_xlabel(method)

    return im, ax


def generate_error_map(fig, target, recon, method, image_ind, rows, cols, relative=False, k=1, kspace=False):
    # Assume rows and cols are available globally
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(rows, cols, image_ind)  # Add to subplot

    # Normalize error between target and reconstruction
    if kspace:
        recon = recon ** 0.4
        target = target ** 0.4

    error = (target - recon) if relative else np.abs(target - recon)
    # normalized_error = error / error.max() if not relative else error
    if relative:
        im = ax.imshow(k * error, cmap='bwr', origin='lower', vmin=-0.0001, vmax=0.0001)  # Plot image
        plt.gca().invert_yaxis()
    else:
        im = ax.imshow(k * error, cmap='jet', vmax=1) if kspace else ax.imshow(k * error, cmap='jet', vmax=0.0001)

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Return plotted image and its axis in the subplot
    return im, ax


def gif_im(true, gen_im, index, type, disc_num=False):
    fig = plt.figure()

    generate_image(fig, true, gen_im, f'z {index}', 1, 2, 1, disc_num=False)
    im, ax = generate_error_map(fig, true, gen_im, f'z {index}', 2, 2, 1)

    plt.savefig(f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN3/gif_{type}_{index - 1}.png')
    plt.close()


def generate_gif(type, num):
    images = []
    for i in range(num):
        images.append(iio.imread(f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN3/gif_{type}_{i}.png'))

    iio.mimsave(f'variation_gif.gif', images, duration=0.25)

    for i in range(num):
        os.remove(f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN3/gif_{type}_{i}.png')


def get_colorbar(fig, im, ax, left=False):
    fig.subplots_adjust(right=0.85)  # Make room for colorbar

    # Get position of final error map axis
    [[x10, y10], [x11, y11]] = ax.get_position().get_points()

    # Appropriately rescale final axis so that colorbar does not effect formatting
    pad = 0.01
    width = 0.01
    cbar_ax = fig.add_axes([x11 + pad, y10, width, y11 - y10]) if not left else fig.add_axes(
        [x10 - 2 * pad, y10, width, y11 - y10])

    cbar = fig.colorbar(im, cax=cbar_ax, format='%.2e')  # Generate colorbar
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.locator_params(nbins=5)

    if left:
        cbar_ax.yaxis.tick_left()
        cbar_ax.yaxis.set_label_position('left')


def get_metrics(args, num_z):
    G = load_best_gan(args)
    G.update_gen_status(val=True)

    losses = {
        'psnr': [],
        'snr': [],
        'ssim': [],
        'apsd': [],
        'mse': [],
        'max_i': []
    }

    means = {
        'psnr': [],
        'snr': [],
        'ssim': [],
        'apsd': [],
        'mse': [],
        'max_i': []
    }

    _, test_loader = create_data_loaders(args, val_only=True, big_test=True) if not args.ls else create_data_loaders_ls(args, val_only=True, big_test=True)

    # compute_cfid.get_cfid(args, G)

    count = 0
    folds = 0
    num_code = num_z

    for i, data in enumerate(test_loader):
        with torch.no_grad():
            y, x, y_true, mean, std = data
            y = y.to(args.device)
            x = x.to(args.device)
            y_true = y_true.to(args.device)

            gens = torch.zeros(size=(y.size(0), num_code, args.in_chans, 384, 384),
                               device=args.device)

            for z in range(num_code):
                gens[:, z, :, :, :] = G(y, y_true)

            avg = torch.mean(gens, dim=1)

            temp_gens = torch.zeros(gens.shape, dtype=gens.dtype)
            for z in range(num_code):
                temp_gens[:, z, :, :, :] = gens[:, z, :, :, :] * std[:, None, None, None].to(args.device) + mean[:,
                                                                                                            None, None,
                                                                                                            None].to(
                    args.device)

            losses['apsd'].append(torch.mean(torch.std(temp_gens, dim=1), dim=(0, 1, 2, 3)).cpu().numpy())

            new_gens = torch.zeros(y.size(0), num_code, 8, 384, 384, 2)
            new_gens[:, :, :, :, :, 0] = temp_gens[:, :, 0:8, :, :]
            new_gens[:, :, :, :, :, 1] = temp_gens[:, :, 8:16, :, :]

            avg_gen = torch.zeros(size=(y.size(0), 8, 384, 384, 2), device=args.device)
            avg_gen[:, :, :, :, 0] = avg[:, 0:8, :, :]
            avg_gen[:, :, :, :, 1] = avg[:, 8:16, :, :]

            gt = torch.zeros(size=(y.size(0), 8, 384, 384, 2), device=args.device)
            gt[:, :, :, :, 0] = x[:, 0:8, :, :]
            gt[:, :, :, :, 1] = x[:, 8:16, :, :]

            for j in range(y.size(0)):
                new_y_true = fft2c_new(ifft2c_new(y_true[j]) * std[j] + mean[j])
                maps = mr.app.EspiritCalib(tensor_to_complex_np(new_y_true.cpu()), calib_width=32,
                                           device=sp.Device(3), show_pbar=False, crop=0.70, kernel_width=6).run().get()
                S = sp.linop.Multiply((384, 384), maps)
                # F = sp.linop.FFT((8, 384, 384), axes=(-1, -2))
                # gt_ksp, avg_ksp = tensor_to_complex_np(fft2c_new(gt[j] * std[j] + mean[j]).cpu()), tensor_to_complex_np(
                #     fft2c_new(avg_gen[j] * std[j] + mean[j]).cpu())
                gt_ksp, avg_ksp = tensor_to_complex_np((gt[j] * std[j] + mean[j]).cpu()), tensor_to_complex_np((avg_gen[j] * std[j] + mean[j]).cpu())

                # avg_gen_np = torch.tensor(S.H * F.H * avg_ksp).abs().numpy()
                avg_gen_np = torch.tensor(S.H * avg_ksp).abs().numpy()

                # gt_np = torch.tensor(S.H * F.H * gt_ksp).abs().numpy()
                gt_np = torch.tensor(S.H * gt_ksp).abs().numpy()


                count += 1

                losses['ssim'].append(ssim(gt_np, avg_gen_np))
                losses['psnr'].append(psnr(gt_np, avg_gen_np))
                losses['snr'].append(snr(gt_np, avg_gen_np))
                # for k in range(num_code):
                #     gen_np = transforms.root_sum_of_squares(
                #         complex_abs(new_gens[j, k, :, :, :, :])).cpu().numpy()
                #     losses['mse'].append(mse(gt_np, gen_np))

                ind = 2
                if i == 0 and j == ind and num_code >= 32:
                    fig = plt.figure()
                    # fig.subplots_adjust(wspace=0, hspace=0.05)

                    generate_image(fig, gt_np, gt_np, f'GT', 1, 3, 2, disc_num=False)
                    generate_image(fig, gt_np, avg_gen_np, f'RC-GAN', 2, 3, 2, disc_num=False)

                    im, ax = generate_error_map(fig, gt_np, avg_gen_np, f'RC-GAN', 4, 3, 2)

                    get_colorbar(fig, im, ax)

                    gen_im_list = []
                    for z in range(num_code):
                        temp_gen = tensor_to_complex_np((new_gens[j, z]).cpu())
                        gen_im_list.append(torch.tensor(S.H * temp_gen).abs().numpy())

                    std_dev = np.zeros(avg_gen_np.shape)
                    for val in gen_im_list:
                        std_dev = std_dev + np.power((val - avg_gen_np), 2)

                    std_dev = std_dev / num_code
                    std_dev = np.sqrt(std_dev)

                    im, ax = generate_image(fig, gt_np, std_dev, 'Std. Dev', 6, 3, 2)

                    get_colorbar(fig, im, ax)

                    plt.savefig(f'comp_plots_{num_code}.png')
                    plt.close(fig)

                    place = 1
                    for r, val in enumerate(gen_im_list):
                        gif_im(gt_np, val, place, 'image')
                        place += 1

                    generate_gif('image', num_code)

                    print("IN PCA")
                    errors = np.zeros((num_code, 384 * 384))
                    count = 0
                    for val in gen_im_list:
                        errors[count, :] = np.abs(gt_np - val).flatten()
                        count += 1

                    print("GOT ERRORS")
                    plt.imshow(np.mean(errors, axis=0).reshape(384, 384))
                    plt.title(f"Mean Error Map")
                    plt.savefig(f"mean_error_map.png")
                    plt.close()

                    errors = errors - np.mean(errors, axis=0)

                    print(f"RANK: {np.linalg.matrix_rank(errors)}")
                    print("GETTING SVD")
                    U, S, Vh = np.linalg.svd(errors, full_matrices=False)

                    print("GOT SVD")
                    lamda = 1 / num_code * S ** 2

                    lamda_flat = lamda
                    print(np.sum(lamda))
                    plt.plot(np.arange(1, len(lamda_flat) + 1, 1), lamda_flat)
                    plt.title(f"Eigenvalues for {num_code} samples")
                    plt.savefig("eigenvalues_pca.png")
                    plt.close()

                if count % 72 == 0:
                    folds += 1
                    means['psnr'].append(losses['psnr'])
                    means['snr'].append(losses['snr'])
                    means['ssim'].append(losses['ssim'])

    # fig, ax1 = plt.subplots(1, 1)
    # fig.suptitle(f'MSE Histogram for {num_code} samples')
    # fig.subplots_adjust(hspace=1)
    # ax1.hist(losses['mse'], bins=20)
    # ax1.set_title('MSE')
    # # ax2.hist(losses['snr'], bins=15)
    # # ax2.set_title('SNR')
    # plt.savefig(f'histo_{num_code}.png')
    # plt.close(fig)
    #
    # fig = plt.figure()
    # fig.suptitle('MSE vs. MAX_I')
    # plt.scatter(losses['max_i'], losses['mse'])
    # plt.xlabel('MAX_I')
    # plt.ylabel('MSE')
    # plt.savefig('mse_v_maxi.png')
    # plt.close(fig)

    print(f'FOLDS: {folds}')
    print(f'PSNR: {np.mean(means["psnr"]):.2f} \\pm {np.std(means["psnr"]) / np.sqrt(folds):.2f}')
    print(f'SNR: {np.mean(means["snr"]):.2f} \\pm {np.std(means["snr"]) / np.sqrt(folds):.2f}')
    print(f'SSIM: {np.mean(means["ssim"]):.4f} \\pm {np.std(means["ssim"]) / np.sqrt(folds):.4f}')
    print(f'APSD: {np.mean(losses["apsd"]):} \\pm {np.std(losses["apsd"]) / np.sqrt(folds):}')

    PSNR_STRING = f'PSNR: {np.mean(means["psnr"]):.2f} \\pm {np.std(means["psnr"]) / np.sqrt(folds):.2f}\n'
    SNR_STRING = f'SNR: {np.mean(means["snr"]):.2f} \\pm {np.std(means["snr"]) / np.sqrt(folds):.2f}\n'
    SSIM_STRING = f'SSIM: {np.mean(means["ssim"]):.4f} \\pm {np.std(means["ssim"]) / np.sqrt(folds):.4f}\n'
    APSD_STRING = f'APSD: {np.mean(losses["apsd"]):} \\pm {np.std(losses["apsd"]) / np.sqrt(folds):}\n'

    send_mail(f"TEST RESULTS - {num_code} code vectors", f'Results\n{PSNR_STRING}{SNR_STRING}{SSIM_STRING}{APSD_STRING}')

    # compute_cfid.get_cfid(args, G)
