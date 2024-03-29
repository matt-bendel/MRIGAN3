import random
import os
import torch

import numpy as np
import torch.autograd as autograd
import matplotlib.pyplot as plt
# TODO: REMOVE
import imageio as iio
import sigpy as sp
import sigpy.mri as mr

################
from typing import Optional
from data import transforms
from evaluation_scripts.metrics import get_metrics
from utils.fftc import fft2c_new, ifft2c_new
from utils.math import complex_abs, tensor_to_complex_np
from utils.parse_args import create_arg_parser
from wrappers.our_gen_wrapper import get_gan, save_model
from data_loaders.prepare_data import create_data_loaders
from data_loaders.prepare_data_ls import create_data_loaders_ls
from torch.nn import functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from mail import send_mail
from evaluation_scripts import compute_cfid

GLOBAL_LOSS_DICT = {
    'g_loss': [],
    'd_loss': [],
    'mSSIM': [],
    'd_acc': []
}


def psnr(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    psnr_val = peak_signal_noise_ratio(gt, pred, data_range=maxval)

    return psnr_val

# def psnr_coil(
#         gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
# ) -> np.ndarray:
#     """Compute Peak Signal to Noise Ratio metric (PSNR)"""
#     if maxval is None:
#         maxval = gt.max()
#         gt = torch.flatten(gt, start_dim=0)
#         pred = torch.flatten(pred, start_dim=0)
#         k = torch.numel(gt)
#     psnr_val = 10*torch.log10((k * maxval**2) / torch.sum((gt-pred)**2) )
#
#     return psnr_val

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


def mssim_tensor(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    # ssim_loss = pytorch_ssim.SSIM()
    return pytorch_msssim.msssim(gt, pred)


def compute_gradient_penalty(D, real_samples, fake_samples, args, y):
    """Calculates the gradient penalty loss for WGAN GP"""
    Tensor = torch.FloatTensor
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(args.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(input=interpolates, y=y)
    if args.patch_disc:
        fake = Tensor(real_samples.shape[0], 1, d_interpolates.shape[-1], d_interpolates.shape[-1]).fill_(1.0).to(args.device)
    else:
        fake = Tensor(real_samples.shape[0], 1).fill_(1.0).to(args.device)

    # Get gradient w.r.t. interpolates
    # print(d_interpolates.shape)
    # print(real_samples.shape)
    # print(fake.shape)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# TODO: REMOVE
def generate_image(fig, target, image, method, image_ind, rows, cols, kspace=False, disc_num=False):
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(rows, cols, image_ind)
    if method != 'GT' and method != 'Std. Dev':
        psnr_val = psnr(target, image)
        snr_val = snr(target, image)
        ssim_val = ssim(target, image)
        if not kspace:
            pred = disc_num
            ax.set_title(
                f'PSNR: {psnr_val:.2f}, SNR: {snr_val:.2f}\nSSIM: {ssim_val:.4f}, Pred: {pred * 100:.2f}% True') if disc_num else ax.set_title(
                f'PSNR: {psnr_val:.2f}, SNR: {snr_val:.2f}\nSSIM: {ssim_val:.4f}')

    if method == 'Std. Dev':
        im = ax.imshow(image, cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        if kspace:
            image = image ** 0.4
            target = target ** 0.4
        im = ax.imshow(np.abs(image), cmap='gray', vmin=0, vmax=np.max(target))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(method)

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


def generate_gif(type):
    images = []
    for i in range(8):
        images.append(iio.imread(f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN3/gif_{type}_{i}.png'))

    iio.mimsave(f'variation_gif.gif', images, duration=0.25)

    for i in range(8):
        os.remove(f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN3/gif_{type}_{i}.png')

def train(args, bl=1, adv_mult=0.0):
    print(f"WEIGHT: {adv_mult}")
    args.exp_dir.mkdir(parents=True, exist_ok=True)

    args.in_chans = 16
    args.out_chans = 16

    G, D, opt_G, opt_D, best_loss, start_epoch = get_gan(args)
    #

    if args.resume:
        start_epoch += 1
    else:
        best_loss = 100000

    train_loader, dev_loader = create_data_loaders(args, big_test=False) if not args.ls else create_data_loaders_ls(args, big_test=False)

    # exit()

    for epoch in range(start_epoch, args.num_epochs):
        batch_loss = {
            'g_loss': [],
            'd_loss': [],
        }

        for i, data in enumerate(train_loader):
            G.update_gen_status(val=False)
            y, x, y_true, mean, std = data
            y = y.to(args.device)
            x = x.to(args.device)
            y_true = y_true.to(args.device)

            for j in range(args.num_iters_discriminator):
                for param in D.parameters():
                    param.grad = None

                x_hat = G(y, y_true)

                real_pred = D(input=x, y=y)
                fake_pred = D(input=x_hat, y=y)

                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(D, x.data, x_hat.data, args, y.data)

                d_loss = fake_pred.mean() - real_pred.mean()
                d_loss += args.gp_weight * gradient_penalty + 0.001 * torch.mean(real_pred ** 2)
                d_loss.backward()
                opt_D.step()

            for param in G.gen.parameters():
                param.grad = None

            gens = torch.zeros(size=(y.size(0), args.num_z, args.in_chans, 384, 384),
                               device=args.device)
            for z in range(args.num_z):
                gens[:, z, :, :, :] = G(y, y_true)

            if args.patch_disc:
                fake_pred = torch.zeros(size=(y.shape[0], args.num_z, 94, 94), device=args.device)
                for k in range(y.shape[0]):
                    cond = torch.zeros(1, gens.shape[2], gens.shape[3], gens.shape[4])
                    cond[0, :, :, :] = y[k, :, :, :]
                    cond = cond.repeat(args.num_z, 1, 1, 1)
                    temp = D(input=gens[k], y=cond)
                    fake_pred[k, :, :, :] = temp[:, 0, :, :]
            else:
                fake_pred = torch.zeros(size=(y.shape[0], args.num_z), device=args.device)
                for k in range(y.shape[0]):
                    cond = torch.zeros(1, gens.shape[2], gens.shape[3], gens.shape[4])
                    cond[0, :, :, :] = y[k, :, :, :]
                    cond = cond.repeat(args.num_z, 1, 1, 1)
                    temp = D(input=gens[k], y=cond)
                    fake_pred[k] = temp[:, 0]

            avg_recon = torch.mean(gens, dim=1)

            gen_pred_loss = torch.mean(fake_pred[0])
            for k in range(y.shape[0] - 1):
                gen_pred_loss += torch.mean(fake_pred[k + 1])

            g_loss = - 1e-3 * gen_pred_loss.mean()
            g_loss += F.mse_loss(avg_recon, x)  # - args.ssim_weight * mssim_tensor(x, avg_recon)
            # g_loss += - std_weight * torch.std(gens, dim=1).mean()

            # if g_loss < -20:
            #     raise Exception

            g_loss.backward()
            opt_G.step()

            batch_loss['g_loss'].append(g_loss.item())
            batch_loss['d_loss'].append(d_loss.item())

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f]"
                % (epoch + 1, args.num_epochs, i, len(train_loader.dataset) / args.batch_size, d_loss.item(),
                   g_loss.item())
            )

        losses = {
            'psnr': [],
            'single_psnr': [],
            'ssim': []
        }

        for i, data in enumerate(dev_loader):
            G.update_gen_status(val=True)
            with torch.no_grad():
                y, x, y_true, mean, std = data
                y = y.to(args.device)
                x = x.to(args.device)
                y_true = y_true.to(args.device)

                gens = torch.zeros(size=(y.size(0), 8, args.in_chans, 384, 384),
                                   device=args.device)
                for z in range(8):
                    gens[:, z, :, :, :] = G(y, y_true, noise_var=1)

                avg = torch.mean(gens, dim=1)

                avg_gen = torch.zeros(size=(y.size(0), 8, 384, 384, 2), device=args.device)
                avg_gen[:, :, :, :, 0] = avg[:, 0:8, :, :]
                avg_gen[:, :, :, :, 1] = avg[:, 8:16, :, :]

                gt = torch.zeros(size=(y.size(0), 8, 384, 384, 2), device=args.device)
                gt[:, :, :, :, 0] = x[:, 0:8, :, :]
                gt[:, :, :, :, 1] = x[:, 8:16, :, :]

                for j in range(y.size(0)):
                    new_y_true = fft2c_new(ifft2c_new(y_true[j]) * std[j] + mean[j])
                    maps = mr.app.EspiritCalib(tensor_to_complex_np(new_y_true.cpu()), calib_width=32,
                                               device=sp.Device(3), show_pbar=False, crop=0.70,
                                               kernel_width=6).run().get()
                    S = sp.linop.Multiply((384, 384), maps)
                    gt_ksp, avg_ksp = tensor_to_complex_np((gt[j] * std[j] + mean[j]).cpu()), tensor_to_complex_np(
                        (avg_gen[j] * std[j] + mean[j]).cpu())

                    avg_gen_np = torch.tensor(S.H * avg_ksp).abs().numpy()
                    gt_np = torch.tensor(S.H * gt_ksp).abs().numpy()

                    single_gen = torch.zeros(8, 384, 384, 2).to(args.device)
                    single_gen[:, :, :, 0] = gens[j, 0, 0:8, :, :]
                    single_gen[:, :, :, 1] = gens[j, 0, 8:16, :, :]

                    single_gen_complex_np = tensor_to_complex_np((single_gen * std[j] + mean[j]).cpu())
                    single_gen_np = torch.tensor(S.H * single_gen_complex_np).abs().numpy()

                    losses['ssim'].append(ssim(gt_np, avg_gen_np))
                    losses['psnr'].append(psnr(gt_np, avg_gen_np))
                    losses['single_psnr'].append(psnr(gt_np, single_gen_np))

                    ind = 1

                    if i == 0 and j == ind:
                        output = transforms.root_sum_of_squares(
                            complex_abs(avg_gen[ind] * std[ind] + mean[ind])).cpu().numpy()
                        target = transforms.root_sum_of_squares(
                            complex_abs(gt[ind] * std[ind] + mean[ind])).cpu().numpy()

                        gen_im_list = []
                        for z in range(8):
                            val_rss = torch.zeros(8, 384, 384, 2).to(args.device)
                            val_rss[:, :, :, 0] = gens[ind, z, 0:8, :, :]
                            val_rss[:, :, :, 1] = gens[ind, z, 8:16, :, :]
                            gen_im_list.append(transforms.root_sum_of_squares(
                                complex_abs(val_rss * std[ind] + mean[ind])).cpu().numpy())

                        std_dev = np.zeros(output.shape)
                        for val in gen_im_list:
                            std_dev = std_dev + np.power((val - output), 2)

                        std_dev = std_dev / 8
                        std_dev = np.sqrt(std_dev)

                        place = 1
                        for r, val in enumerate(gen_im_list):
                            gif_im(target, val, place, 'image')
                            place += 1

                        generate_gif('image')

                        fig = plt.figure()
                        ax = fig.add_subplot(1, 1, 1)
                        im = ax.imshow(std_dev, cmap='viridis')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        fig.subplots_adjust(right=0.85)  # Make room for colorbar

                        # Get position of final error map axis
                        [[x10, y10], [x11, y11]] = ax.get_position().get_points()

                        pad = 0.01
                        width = 0.02
                        cbar_ax = fig.add_axes([x11 + pad, y10, width, y11 - y10])

                        fig.colorbar(im, cax=cbar_ax)

                        plt.savefig(f'std_dev_gen.png')
                        plt.close()

        psnr_loss = np.mean(losses['psnr'])
        CFID = compute_cfid.get_cfid(args, G, dev_loader)

        best_model = CFID < best_loss
        best_loss = CFID if best_model else best_loss

        GLOBAL_LOSS_DICT['g_loss'].append(np.mean(batch_loss['g_loss']))
        GLOBAL_LOSS_DICT['d_loss'].append(np.mean(batch_loss['d_loss']))

        save_str = f"END OF EPOCH {epoch + 1}: [Average D loss: {GLOBAL_LOSS_DICT['d_loss'][epoch - start_epoch]:.4f}] [Average G loss: {GLOBAL_LOSS_DICT['g_loss'][epoch - start_epoch]:.4f}]\n"
        print(save_str)
        save_str_2 = f"[Avg PSNR: {np.mean(losses['psnr']):.2f}] [Avg SSIM: {np.mean(losses['ssim']):.4f}]"
        print(save_str_2)

        send_mail(f"EPOCH {epoch + 1} UPDATE", f"Metrics:\nPSNR: {np.mean(losses['psnr']):.2f}\nSSIM: {np.mean(losses['ssim']):.4f}\nCFID: {CFID}", file_name="variation_gif.gif")

        # if adv_weight < 1.5 and psnr_diff > 0:
        #     save_model(args, epoch, G.gen, opt_G, best_loss, best_model, 'generator')
        #     save_model(args, epoch, D, opt_D, best_loss, best_model, 'discriminator')
        # elif adv_weight > 1.4:
        save_model(args, epoch, G.gen, opt_G, best_loss, best_model, 'generator')
        save_model(args, epoch, D, opt_D, best_loss, best_model, 'discriminator')



if __name__ == '__main__':
    cuda = True if torch.cuda.is_available() else False
    torch.backends.cudnn.benchmark = True

    args = create_arg_parser().parse_args()
    # restrict visible cuda devices
    if args.data_parallel or (args.device >= 0):
        if not args.data_parallel:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    args.in_chans = 16
    args.out_chans = 16

    vals = [1e-3]
    args.checkpoint_dir = "/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN3/trained_models"
    for val in vals:
        try:
            train(args, bl=0, adv_mult=val)
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(e)
            send_mail("TRAINING CRASH", "See terminal for failure cause.")

        try:
            for i in range(6):
                num = 2 ** i
                get_metrics(args, num, is_super=True, std_val=val)
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(e)
            send_mail("TESTING FAILED", "See terminal for failure cause.")
