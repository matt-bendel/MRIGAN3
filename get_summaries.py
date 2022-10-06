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
from torchsummary import summary
from models.generators.our_gen import GeneratorModel
from models.discriminators.patch_disc import PatchDisc

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

    G = GeneratorModel(
        in_chans=args.in_chans + 2,
        out_chans=args.out_chans,
    ).to(torch.device('cuda'))

    D = PatchDisc(
        input_nc=args.in_chans * 2
    ).to(torch.device('cuda'))

    summary(G, (18, 384, 384))
    summary(D, (32, 384, 384))

    exit()