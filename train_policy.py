import random
import os
import torch
import time
import pathlib
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
from wrappers.our_gen_wrapper import load_best_gan
from models.policy.policy_model import PolicyModel

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


def generate_gif(type, tot):
    images = []
    for i in range(tot):
        images.append(iio.imread(f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN3/gif_{type}_{i}.png'))

    iio.mimsave(f'variation_gif.gif', images, duration=0.25)

    for i in range(tot):
        os.remove(f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN3/gif_{type}_{i}.png')

def compute_scores(G, kspace, mask, zf, gt_mean, gt_std):
    recons = torch.zeros(kspace.size(0), 8, 8, 384, 384, 2).cuda()
    print(kspace.shape)
    normalized_kspace = fft2c_new((ifft2c_new(kspace) - gt_mean[:, None, None, None]) / gt_std[:, None, None, None])

    for z in range(8):
        recon = G(zf, normalized_kspace, mask=mask)
        recon = recon * gt_std[:, None, None, None] + gt_mean[:, None, None, None]
        recons[:, z, :, :, :, 0] = recon[:, 0:8, :, :]
        recons[:, z, :, :, :, 1] = recon[:, 8:16, :, :]

    kspace_recons = fft2c_new(recons)

    kspace_var = torch.var(complex_abs(kspace_recons), dim=1)
    var = torch.mean(kspace_var, dim=(1, 2, 3))

    return kspace_recons, var

def get_policy_probs(model, recons, mask, num_traj=8):
    res = mask.size(-2)
    # Reshape trajectory dimension into batch dimension for parallel forward pass
    # Obtain policy model logits
    output = model(recons)
    # Reshape trajectories back into their own dimension
    output = output.view(mask.size(0), 1, res)
    # Mask already acquired rows by setting logits to very negative numbers
    loss_mask = (mask == 0)[:, 0, 0, :, 0].unsqueeze(1)
    logits = torch.where(loss_mask.byte(), output, -1e7 * torch.ones_like(output))
    # Softmax over 'logits' representing row scores
    probs = torch.nn.functional.softmax(logits - logits.max(dim=-1, keepdim=True)[0], dim=-1)
    # Also need this for sampling the next row at the end of this loop
    policy = torch.distributions.Categorical(probs)
    return policy, probs

def train(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    num_traj = 8
    G = load_best_gan(args)
    G.update_gen_status(val=True)
    for param in G.gen.parameters():
        param.requires_grad = False

    # TODO: Load on resume
    # Improvement model to train
    model = PolicyModel(
        resolution=384,
        in_chans=16,
        chans=16,
        num_pool_layers=4,
        drop_prob=0,
        fc_size=1024,
    ).to(args.device)

    # Add mask parameters for training
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    optimiser = torch.optim.Adam(model.parameters(), 5e-5, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimiser, 40, 0.1)
    start_epoch = 0

    if args.resume:
        start_epoch += 1

    train_loader, dev_loader = create_data_loaders(args, big_test=False) if not args.ls else create_data_loaders_ls(args, big_test=False)

    for epoch in range(start_epoch, 50):
        model.train()
        for i, data in enumerate(train_loader):
            break
            epoch_start = time.time()
            zf, gt, kspace, gt_mean, gt_std, mask = data
            zf = zf.cuda()
            gt = gt.cuda()
            kspace = kspace.cuda()
            gt_mean = gt_mean.cuda()
            gt_std = gt_std.cuda()
            mask = mask.cuda()

            optimiser.zero_grad()
            recons, base_score = compute_scores(G, kspace, mask, zf, gt_mean, gt_std)
            base_score = base_score.unsqueeze(1).repeat(1, num_traj)

            accum_loss = 0
            for step in range(48):
                # print(f"STEP: {step+1}")
                # Get policy and probabilities.
                policy_in = torch.zeros(recons.size(0), 16, 384, 384).cuda()
                var_recons = torch.var(recons, dim=1)
                policy_in[:, 0:8, :, :] = var_recons[:, :, :, :, 0]
                policy_in[:, 8:16, :, :] = var_recons[:, :, :, :, 1]

                policy, probs = get_policy_probs(model, policy_in, mask)
                actions = torch.multinomial(probs.squeeze(1), num_traj, replacement=True)
                actions = actions.unsqueeze(1)  # batch x num_traj -> batch x 1 x num_traj
                # probs shape = batch x 1 x res
                action_logprobs = torch.log(torch.gather(probs, -1, actions)).squeeze(1)
                actions = actions.squeeze(1)

                mask = mask.unsqueeze(1).repeat(1, num_traj, 1, 1, 1, 1)
                for j in range(actions.size(0)):
                    for k in range(actions.size(1)):
                        mask[j, k, :, :, actions[j,k], :] = 1

                var_scores = torch.zeros(mask.size(0), mask.size(1)).cuda()
                for k in range(mask.size(1)):
                    m = mask[:, k, :, :, :, :]
                    recons = (1-m.unsqueeze(1).repeat(1, 8, 1, 1, 1, 1))*recons + m.unsqueeze(1).repeat(1, 8, 1, 1, 1, 1)*kspace.unsqueeze(1).repeat(1, 8, 1, 1, 1, 1)
                    var_scores[:, k] = torch.mean(torch.var(complex_abs(recons), dim=1), dim=(1, 2, 3))

                # batch x num_trajectories
                action_rewards = base_score - var_scores
                base_score = var_scores
                # batch x 1
                avg_reward = torch.mean(action_rewards, dim=-1, keepdim=True)
                # Store for non-greedy model (we need the full return before we can do a backprop step)
                # action_list.append(actions)
                # logprob_list.append(action_logprobs)
                # reward_list.append(action_rewards)

                # Local baseline
                loss = -1 * (action_logprobs * (action_rewards - avg_reward)) / (actions.size(-1) - 1)
                loss = loss.sum(dim=1)
                # Average over batch
                # Divide by batches_step to mimic taking mean over larger batch
                loss = loss.mean()  # For consistency: we generally set batches_step to 1 for greedy
                loss.backward()
                accum_loss += loss.item()

                idx = random.randint(0, mask.shape[1] - 1)
                mask = mask[:, idx:idx + 1, :, :, :, :].squeeze(1)
                recons = (1-mask.unsqueeze(1).repeat(1, 8, 1, 1, 1, 1))*recons + mask.unsqueeze(1).repeat(1, 8, 1, 1, 1, 1)*kspace.unsqueeze(1).repeat(1, 8, 1, 1, 1, 1)

            epoch_end_hr = time.time() - epoch_start
            # epoch_end_hr /= 60 # minutes
            # epoch_end_hr /= 60 # hrs
            print(f"EPOCH: {epoch+1} | BATCH: {i+1}/{len(train_loader)} | TIME: {epoch_end_hr:.2f}s | LOSS: {accum_loss}")
            optimiser.step()

        # TODO: This, one full sampling trajectory for arbitrary batch element
        ind = 2
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(dev_loader):
                zf, gt, kspace, gt_mean, gt_std, mask = data
                zf = zf.cuda()
                gt = gt.cuda()
                kspace = kspace.cuda()
                gt_mean = gt_mean.cuda()
                gt_std = gt_std.cuda()
                mask = mask.cuda()
                recons, base_score = compute_scores(G, kspace, mask, zf, gt_mean, gt_std)
                mean_kspace = torch.mean(recons, dim=1)[ind]
                maps = mr.app.EspiritCalib(tensor_to_complex_np(mean_kspace.cpu()), calib_width=16,
                                           device=sp.Device(3), show_pbar=False, crop=0.70,
                                           kernel_width=6).run().get()
                S = sp.linop.Multiply((384, 384), maps)

                target = torch.zeros(size=(8, 384, 384, 2), device=args.device)
                target[:, :, :, 0] = gt[ind, 0:8, :, :]
                target[:, :, :, 1] = gt[ind, 8:16, :, :]
                target = tensor_to_complex_np((target * gt_std[ind] + gt_mean[ind]).cpu())
                target_np = torch.tensor(S.H * target).abs().numpy()

                im_list = []
                for step in range(48):
                    print(f"STEP: {step+1}")
                    im_recon = torch.tensor(S.H * tensor_to_complex_np(ifft2c_new(torch.mean(recons, dim=1)[ind]).cpu())).abs().numpy()
                    im_list.append(im_recon)

                    policy_in = torch.zeros(recons.size(0), 16, 384, 384).cuda()
                    var_recons = torch.var(recons, dim=1)
                    policy_in[:, 0:8, :, :] = var_recons[:, :, :, :, 0]
                    policy_in[:, 8:16, :, :] = var_recons[:, :, :, :, 1]

                    policy, probs = get_policy_probs(model, policy_in, mask, num_traj=1)
                    if step == 0:
                        actions = torch.multinomial(probs.squeeze(1), 1, replacement=True)
                    else:
                        actions = policy.sample()

                    for j in range(mask.size(0)):
                        mask[j, :, :, actions[j, 0], :] = 1

                    recons = (1 - mask.unsqueeze(1).repeat(1, 8, 1, 1, 1, 1)) * recons + mask.unsqueeze(1).repeat(1, 8, 1, 1, 1, 1) \
                             * kspace.unsqueeze(1).repeat(1, 8, 1, 1, 1, 1)

                place = 1
                for r, val in enumerate(im_list):
                    gif_im(target_np, val, place, 'image')
                    place += 1

                generate_gif('image', len(im_list))
                exit()
                break

        # scheduler.step()

        send_mail(f"POLICY EPOCH {epoch + 1} UPDATE", f"TODO: FILL OUT", file_name="variation_gif.gif")

        torch.save(
            {
                'epoch': epoch,
                'args': args,
                'model': model.state_dict(),
                'optimizer': optimiser.state_dict(),
            },
            f=pathlib.Path('/home/bendel.8/Git_Repos/MRIGAN3/trained_models/policy') / 'policy_model.pt'
        )

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

    args.batch_size = 4
    args.in_chans = 16
    args.out_chans = 16

    args.checkpoint_dir = "/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN3/trained_models"
    train(args)
