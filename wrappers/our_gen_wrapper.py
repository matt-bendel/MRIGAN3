import pathlib
import shutil
import torch
import numpy as np

from utils.fftc import ifft2c_new, fft2c_new
from utils.get_mask import get_mask
from torch.nn.parallel import DistributedDataParallel as DDP

# THIS FILE CONTAINTS UTILITY FUNCTIONS FOR OUR GAN AND A WRAPPER CLASS FOR THE GENERATOR
# TODO: CHANGE BACK TO LOADING BEST
def load_best_gan(args):
    from utils.prepare_models import build_model, build_model_sg
    checkpoint_file_gen = pathlib.Path(
        f'{args.checkpoint_dir}/generator_best_model.pt')
    # checkpoint_file_gen = pathlib.Path(
    #     f'{args.checkpoint_dir}/generator_model.pt')
    checkpoint_gen = torch.load(checkpoint_file_gen, map_location=torch.device('cuda'))

    generator = build_model(args) if not args.stylegan else build_model_sg(args)
    if args.data_parallel:
        generator = torch.nn.DataParallel(generator)

    generator.load_state_dict(checkpoint_gen['model'])
    print(f'EPOCH: {checkpoint_gen["epoch"]}')

    generator = GANWrapper(generator, args)

    return generator


def get_gan(args, rank=0, world_size=2):
    from utils.prepare_models import build_model, build_model_sg, build_optim, build_discriminator

    if args.resume:
        checkpoint_file_gen = pathlib.Path(
            f'{args.checkpoint_dir}/generator_model.pt')
        checkpoint_gen = torch.load(checkpoint_file_gen, map_location=torch.device('cuda'))

        checkpoint_file_dis = pathlib.Path(
            f'{args.checkpoint_dir}/discriminator_model.pt')
        checkpoint_dis = torch.load(checkpoint_file_dis, map_location=torch.device('cuda'))

        generator = build_model(args) if not args.stylegan else build_model_sg(args)
        discriminator = build_discriminator(args)

        if args.data_parallel:
            generator = torch.nn.DataParallel(generator)
            discriminator = torch.nn.DataParallel(discriminator)

        generator.load_state_dict(checkpoint_gen['model'])

        generator = GANWrapper(generator, args)

        opt_gen = build_optim(args, generator.gen.parameters())
        opt_gen.load_state_dict(checkpoint_gen['optimizer'])

        discriminator.load_state_dict(checkpoint_dis['model'])

        opt_dis = build_optim(args, discriminator.parameters())
        opt_dis.load_state_dict(checkpoint_dis['optimizer'])

        best_loss = checkpoint_gen['best_dev_loss']
        start_epoch = checkpoint_gen['epoch']

    else:
        generator = build_model(args) if not args.stylegan else build_model_sg(args)
        discriminator = build_discriminator(args)

        if args.data_parallel:
            generator = DDP(generator.to(rank), device_ids=[rank], output_device=rank)#torch.nn.DataParallel(generator)
            discriminator = DDP(discriminator.to(rank), device_ids=[rank], output_device=rank)#torch.nn.DataParallel(discriminator)

        generator = GANWrapper(generator, args)

        # Optimizers
        opt_gen = torch.optim.Adam(generator.gen.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))
        opt_dis = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))
        best_loss = 0
        start_epoch = 0

    return generator, discriminator, opt_gen, opt_dis, best_loss, start_epoch


def save_model(args, epoch, model, optimizer, best_dev_loss, is_new_best, m_type):
    fpath = args.exp_dir / 'ls' if args.ls else args.exp_dir if not args.stylegan else args.exp_dir / 'stylegan'
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': args.exp_dir
        },
        f=fpath / f'{m_type}_model.pt'
    )

    if is_new_best:
        shutil.copyfile(fpath / f'{m_type}_model.pt',
                        fpath / f'{m_type}_best_model.pt')

def save_model_ddp(args, epoch, model, optimizer, best_dev_loss, is_new_best, m_type, rank):
    fpath = args.exp_dir / 'ls' if args.ls else args.exp_dir if not args.stylegan else args.exp_dir / 'stylegan'
    if rank == 0:
        torch.save(
            {
                'epoch': epoch,
                'args': args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_dev_loss': best_dev_loss,
                'exp_dir': args.exp_dir
            },
            f=fpath / f'{m_type}_model.pt'
        )

        if is_new_best:
            shutil.copyfile(fpath / f'{m_type}_model.pt',
                            fpath / f'{m_type}_best_model.pt')

    dist.barrier()
    # map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    # ddp_model.load_state_dict(
    #     torch.load(fpath / f'{m_type}_model.pt', map_location=map_location))

class GANWrapper:
    def __init__(self, gen, args):
        self.args = args
        self.resolution = args.im_size
        self.gen = gen
        self.data_consistency = True

    def get_noise(self, num_vectors, var, mask):
        # return torch.cuda.FloatTensor(np.random.normal(size=(num_vectors, self.args.latent_size), scale=1))
        z = torch.randn(num_vectors, self.resolution, self.resolution, 2).cuda()
        noise_fft = fft2c_new(z)
        measured_noise = ifft2c_new(mask[:, 0, :, :, :] * noise_fft).permute(0, 3, 1, 2)
        # nonmeasured_noise = ifft2c_new((1 - mask[:, 0, :, :, :]) * noise_fft).permute(0, 3, 1, 2)
        return measured_noise

        # return torch.cat([measured_noise, nonmeasured_noise], dim=1)
        # return torch.randn(num_vectors, 2, self.resolution, self.resolution).cuda()
        # return torch.randn(num_vectors, 2, self.resolution, self.resolution).cuda()

    def update_gen_status(self, val):
        self.gen.eval() if val else self.gen.train()

    def reformat(self, samples):
        reformatted_tensor = torch.zeros(size=(samples.size(0), 8, self.resolution, self.resolution, 2),
                                         device=self.args.device)
        reformatted_tensor[:, :, :, :, 0] = samples[:, 0:8, :, :].clone()
        reformatted_tensor[:, :, :, :, 1] = samples[:, 8:16, :, :].clone()

        return reformatted_tensor

    def readd_measures(self, samples, measures, mask):
        reformatted_tensor = self.reformat(samples)
        reconstructed_kspace = fft2c_new(reformatted_tensor)

        reconstructed_kspace = mask * measures + (1 - mask) * reconstructed_kspace

        # for i in range(reconstructed_kspace.size(0)):
        #     reconstructed_kspace[i, :, inds[i, 0], inds[i, 1], :] = measures[i, :, inds[i, 0], inds[i, 1], :]

        image = ifft2c_new(reconstructed_kspace)

        output_im = torch.zeros(size=samples.shape, device=self.args.device)
        output_im[:, 0:8, :, :] = image[:, :, :, :, 0].clone()
        output_im[:, 8:16, :, :] = image[:, :, :, :, 1].clone()

        return output_im

    def __call__(self, y, true_measures, noise_var=1, mask=None, inds=None):
        num_vectors = y.size(0)
        z = self.get_noise(num_vectors, 1, mask)
        samples = self.gen(input=torch.cat([y, z], dim=1), mid_z=None)
        # samples = self.gen(y, None, [torch.randn(y.size(0), 512, device=y.device)], return_latents=False, truncation=None, truncation_latent=None)
        #
        samples = self.readd_measures(samples, true_measures, mask)
        return samples
