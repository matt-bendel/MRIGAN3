# import required module
import os
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
from utils.fftc import fft2c_new, ifft2c_new
from utils.math import complex_abs, tensor_to_complex_np
from torch.utils.data import DataLoader

from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils.parse_args import create_arg_parser
from wrappers.our_gen_wrapper import load_best_gan

def generate_image(fig, target, image, method, image_ind, rows, cols, kspace=False, disc_num=False):
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(rows, cols, image_ind)
    if method != 'GT' and method != 'Std. Dev':
        psnr_val = psnr(target, image)
        snr_val = snr(target, image)
        ssim_val = ssim(target, image)
        if method != None:
            ax.set_title(method, size=10)

        # padding = 5
        # ax.annotate(
        #     s=f'PSNR: {psnr_val:.2f}\nSNR: {snr_val:.2f}\nSSIM: {ssim_val:.4f}',
        #     fontsize='xx-small',
        #     xy=(0, 0),
        #     xytext=(padding - 1, -(padding - 1)),
        #     textcoords='offset pixels',
        #     # bbox=dict(facecolor='white', alpha=1, pad=padding),
        #     va='top',
        #     ha='right',
        # )
        ax.text(1, 0.8, f'PSNR: {psnr_val:.2f}\nSNR: {snr_val:.2f}\nSSIM: {ssim_val:.4f}', transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='center', fontsize='xx-small', color='yellow')

    if method == 'Std. Dev':
        im = ax.imshow(image, cmap='viridis', vmin=0, vmax=5e-5)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        if kspace:
            image = image ** 0.4
            target = target ** 0.4
        ax.set_title(method, size=10)
        im = ax.imshow(np.abs(image), cmap='gray', vmin=0, vmax=np.max(target))
        ax.set_xticks([])
        ax.set_yticks([])

    return im, ax


def generate_error_map(fig, target, recon, image_ind, rows, cols, relative=False, k=1, kspace=False, title=None):
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
        im = ax.imshow(k * error, cmap='jet', vmax=1) if kspace else ax.imshow(k * error, cmap='jet', vmin=0, vmax=0.0001)

    if title != None:
        ax.set_title(title, size=10)
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Return plotted image and its axis in the subplot
    return im, ax


def get_colorbar(fig, im, ax, left=False):
    fig.subplots_adjust(right=0.85)  # Make room for colorbar

    # Get position of final error map axis
    [[x10, y10], [x11, y11]] = ax.get_position().get_points()

    # Appropriately rescale final axis so that colorbar does not effect formatting
    pad = 0.01
    width = 0.01
    cbar_ax = fig.add_axes([x11 + pad, y10, width, y11 - y10]) if not left else fig.add_axes([x10 - 2*pad, y10, width, y11 - y10])

    cbar = fig.colorbar(im, cax=cbar_ax, format='%.2e')  # Generate colorbar
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.locator_params(nbins=5)

    if left:
        cbar_ax.yaxis.tick_left()
        cbar_ax.yaxis.set_label_position('left')

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

def create_mean_error_plots(avg, std_devs, gt, plot_num):
    num_rows = 3
    num_cols = 4

    fig = plt.figure()
    fig.subplots_adjust(wspace=0, hspace=0.05)
    generate_image(fig, gt['ours'], gt['ours'], 'GT', 1, num_rows, num_cols)

    labels = ['Ours', 'Adler', 'Langevin']
    im_er, ax_er = None, None
    im_std, ax_std = None, None

    avg_keys = ['ours', 'adler', 'langevin']
    for i in range(num_cols - 1):
        generate_image(fig, gt, avg[avg_keys[i]], labels[i], i + 1, num_rows, num_cols)
        if i == 0:
            im_er, ax_er = generate_error_map(fig, gt, avg[avg_keys[i]], i + 5, num_rows, num_cols)
            im_std, ax_std = generate_image(fig, gt, std_devs[avg_keys[i]], 'Std. Dev', i + 9, num_rows, num_cols)
        else:
            generate_error_map(fig, gt, avg[avg_keys[i]], i + 5, num_rows, num_cols)
            generate_image(fig, gt, std_devs[avg_keys[i]], 'Std. Dev', i + 9, num_rows, num_cols)

    get_colorbar(fig, im_er, ax_er, left=True)
    get_colorbar(fig, im_std, ax_std, left=True)

    plt.savefig(f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN3/asilomar_plots/mean_error_{plot_num}.png', bbox_inches='tight')

def main(args):
    args.batch_size = 1
    ref_directory = '/storage/fastMRI_brain/data/small_T2_test'
    # iterate over files in
    # that directory

    args.checkpoint_dir = "/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN3/trained_models/asilomar_ours"
    G_ours = load_best_gan(args)
    G_ours.update_gen_status(val=True)

    args.checkpoint_dir = "/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN3/trained_models/asilomar_adler"
    G_adler = load_best_gan(args)
    G_adler.update_gen_status(val=True)

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
        num_workers=16,
        pin_memory=True
    )

    num_code = 32

    for i, data in enumerate(loader):
        with torch.no_grad():
            y, x, y_true, mean, std, fname, slice = data
            y = y.to(args.device)
            x = x.to(args.device)
            y_true = y_true.to(args.device)

            gens_ours = torch.zeros(size=(num_code, args.in_chans, 384, 384),
                               device=args.device)
            gens_adler = torch.zeros(size=(num_code, args.in_chans, 384, 384),
                                    device=args.device)

            for z in range(num_code):
                gens_ours[z, :, :, :] = G_ours(y, y_true)[0]
                gens_adler[z, :, :, :] = G_adler(y, y_true)[0]

            print(gens_ours.shape)

            avg_ours = torch.mean(gens_ours, dim=0)
            avg_adler = torch.mean(gens_adler, dim=0)

            print(avg_ours.shape)

            temp_gens_ours = torch.zeros(gens_ours.shape, dtype=gens_ours.dtype)
            temp_gens_adler = torch.zeros(gens_adler.shape, dtype=gens_adler.dtype)
            for z in range(num_code):
                temp_gens_ours[z, :, :, :] = gens_ours[z, :, :, :] * std[0].to(args.device) + mean[0].to(args.device)
                temp_gens_adler[z, :, :, :] = gens_adler[z, :, :, :] * std[0].to(args.device) + mean[0].to(args.device)

            new_gens_ours = torch.zeros(num_code, 8, 384, 384, 2)
            new_gens_ours[:, :, :, :, 0] = temp_gens_ours[:, 0:8, :, :]
            new_gens_ours[:, :, :, :, 1] = temp_gens_ours[:, 8:16, :, :]

            new_gens_adler = torch.zeros(num_code, 8, 384, 384, 2)
            new_gens_adler[:, :, :, :, 0] = temp_gens_adler[:, 0:8, :, :]
            new_gens_adler[:, :, :, :, 1] = temp_gens_adler[:, 8:16, :, :]

            avg_gen_ours = torch.zeros(size=(8, 384, 384, 2), device=args.device)
            avg_gen_ours[:, :, :, 0] = avg_ours[0:8, :, :]
            avg_gen_ours[:, :, :, 1] = avg_ours[8:16, :, :]

            avg_gen_adler = torch.zeros(size=(8, 384, 384, 2), device=args.device)
            avg_gen_adler[:, :, :, 0] = avg_adler[0:8, :, :]
            avg_gen_adler[:, :, :, 1] = avg_adler[8:16, :, :]

            gt = torch.zeros(size=(8, 384, 384, 2), device=args.device)
            gt[:, :, :, 0] = x[0, 0:8, :, :]
            gt[:, :, :, 1] = x[0, 8:16, :, :]

            new_y_true = fft2c_new(ifft2c_new(y_true[0]) * std[0] + mean[0])
            maps = mr.app.EspiritCalib(tensor_to_complex_np(new_y_true.cpu()), calib_width=32,
                                       device=sp.Device(3), show_pbar=False, crop=0.70,
                                       kernel_width=6).run().get()
            S = sp.linop.Multiply((384, 384), maps)
            gt_ksp, avg_ksp_ours, avg_ksp_adler = tensor_to_complex_np((gt * std[0] + mean[0]).cpu()), tensor_to_complex_np(
                (avg_gen_ours * std[0] + mean[0]).cpu()), tensor_to_complex_np(
                (avg_gen_adler * std[0] + mean[0]).cpu())

            avg_gen_np_ours = torch.tensor(S.H * avg_ksp_ours).abs().numpy()
            avg_gen_np_adler = torch.tensor(S.H * avg_ksp_adler).abs().numpy()

            gt_np = torch.tensor(S.H * gt_ksp).abs().numpy()

            ours_samples_np = np.zeros((num_code, 384, 384))
            adler_samples_np = np.zeros((num_code, 384, 384))

            for z in range(num_code):
                gen_ours = tensor_to_complex_np((new_gens_ours[z] * std[0] + mean[0]).cpu())
                gen_adler = tensor_to_complex_np((new_gens_adler[z] * std[0] + mean[0]).cpu())

                ours_samples_np[z] = torch.tensor(S.H * gen_ours).abs().numpy()
                adler_samples_np[z] = torch.tensor(S.H * gen_adler).abs().numpy()

            std_ours_np = np.std(ours_samples_np, axis=0)
            std_adler_np = np.std(adler_samples_np, axis=0)

            recon_directory = f'/storage/fastMRI_brain/Langevin_Recons_R=4/'
            langevin_recons = np.zeros((32, 384, 384))
            recon_object = None

            for j in range(num_code):
                try:
                    new_filename = recon_directory + filename + f'|langevin|slide_idx_{slice}_R=4_sample={j}_outputs.pt'
                    recon_object = torch.load(new_filename)
                except:
                    proint(new_filename)
                    exceptions = True
                    continue
                # temp_recon = unnormalize(recon_object['mvue'], recon_object['zfr'])

                langevin_recons[j] = complex_abs(recon_object['mvue'][0].permute(1, 2, 0)).cpu().numpy()

            if exceptions:
                print("EXCEPTION\n")
                exceptions = False
                continue

            langevin_avg = np.mean(recons, axis=0)
            langevin_gt = recon_object['gt'][0][0].abs().cpu().numpy()
            langevin_std = np.std(langevin_recons, axis=0)

            std_dict = {
                'ours': std_ours_np,
                'adler': std_adler_np,
                'langevin': langevin_std
            }

            avg_dict = {
                'ours': avg_gen_np_ours,
                'adler': avg_gen_np_adler,
                'langevin': langevin_avg
            }

            gt_dict = {
                'ours': gt_np,
                'adler': gt_np,
                'langevin': langevin_gt
            }

            create_mean_error_plots(avg_dict, std_dict, gt_dict, i)

            exit()



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

    main(args)