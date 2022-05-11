import torch

from models.generators.our_gen import GeneratorModel
from models.discriminators.our_disc import DiscriminatorModel


def build_model(args):
    model = GeneratorModel(
        in_chans=args.in_chans + 2,
        out_chans=args.out_chans,
    ).to(torch.device('cuda'))

    return model


def build_discriminator(args):
    model = DiscriminatorModel(
        in_chans=args.in_chans * 2,
        out_chans=args.out_chans,
    ).to(torch.device('cuda'))

    return model


def build_optim(args, params):
    return torch.optim.Adam(params, lr=args.lr, betas=(args.beta_1, args.beta_2))
