import torch

from models.generators.our_gen import GeneratorModel
from models.generators.stylegan import StyleGAN
from models.discriminators.our_disc import DiscriminatorModel
from models.discriminators.patch_disc import PatchDisc


def build_model(args):
    model = GeneratorModel(
        in_chans=args.in_chans,
        out_chans=args.out_chans,
    ).to(torch.device('cuda'))

    return model


def build_model_sg(args):
    model = StyleGAN().to(torch.device('cuda'))

    return model


def build_discriminator(args):
    if args.patch_disc:
        print('PATCH DISC')
        model = PatchDisc(
            input_nc=args.in_chans * 2,
        ).to(torch.device('cuda'))
    else:
        model = DiscriminatorModel(
            in_chans=args.in_chans * 2,
            out_chans=args.out_chans,
        ).to(torch.device('cuda'))

    return model


def build_optim(args, params):
    return torch.optim.Adam(params, lr=args.lr, betas=(args.beta_1, args.beta_2))
