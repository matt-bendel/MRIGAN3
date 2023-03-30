# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import torch

import numpy as np
import sigpy as sp
from scipy import linalg
from utils.math import complex_abs

from tqdm import tqdm
import torchvision.transforms as transforms


def symmetric_matrix_square_root_torch(mat, eps=1e-10):
    """Compute square root of a symmetric matrix.
    Note that this is different from an elementwise square root. We want to
    compute M' where M' = sqrt(mat) such that M' * M' = mat.
    Also note that this method **only** works for symmetric matrices.
    Args:
      mat: Matrix to take the square root of.
      eps: Small epsilon such that any element less than eps will not be square
        rooted to guard against numerical instability.
    Returns:
      Matrix square root of mat.
    """
    # Unlike numpy, tensorflow's return order is (s, u, v)
    u, s, v = torch.linalg.svd(mat)
    # sqrt is unstable around 0, just use 0 in such case
    si = s
    si[torch.where(si >= eps)] = torch.sqrt(si[torch.where(si >= eps)])

    # Note that the v returned by Tensorflow is v = V
    # (when referencing the equation A = U S V^T)
    # This is unlike Numpy which returns v = V^T
    return torch.matmul(torch.matmul(u, torch.diag(si)), v)


def trace_sqrt_product_torch(sigma, sigma_v):
    """Find the trace of the positive sqrt of product of covariance matrices.
    '_symmetric_matrix_square_root' only works for symmetric matrices, so we
    cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
    ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).
    Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
    We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
    Note the following properties:
    (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
      => eigenvalues(A A B B) = eigenvalues (A B B A)
    (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
      => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
    (iii) forall M: trace(M) = sum(eigenvalues(M))
      => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                    = sum(sqrt(eigenvalues(A B B A)))
                                    = sum(eigenvalues(sqrt(A B B A)))
                                    = trace(sqrt(A B B A))
                                    = trace(sqrt(A sigma_v A))
    A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
    use the _symmetric_matrix_square_root function to find the roots of these
    matrices.
    Args:
      sigma: a square, symmetric, real, positive semi-definite covariance matrix
      sigma_v: same as sigma
    Returns:
      The trace of the positive square root of sigma*sigma_v
    """

    # Note sqrt_sigma is called "A" in the proof above
    sqrt_sigma = symmetric_matrix_square_root_torch(sigma)

    # This is sqrt(A sigma_v A) above
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))

    return torch.trace(symmetric_matrix_square_root_torch(sqrt_a_sigmav_a))


# **Estimators**
#
def sample_covariance_torch(a, b):
    '''
    Sample covariance estimating
    a = [N,m]
    b = [N,m]
    '''
    assert (a.shape[0] == b.shape[0])
    assert (a.shape[1] == b.shape[1])
    m = a.shape[1]
    N = a.shape[0]
    return torch.matmul(torch.transpose(a, 0, 1), b) / N


class CFIDMetric:
    """Helper function for calculating CFID metric.

    Note: This code is adapted from Facebook's FJD implementation in order to compute
    CFID in a streamlined fashion.

    Args:
        gan: Model that takes in a conditioning tensor and yields image samples.
        reference_loader: DataLoader that yields (images, conditioning) pairs
            to be used as the reference distribution.
        condition_loader: Dataloader that yields (image, conditioning) pairs.
            Images are ignored, and conditions are fed to the GAN.
        image_embedding: Function that takes in 4D [B, 3, H, W] image tensor
            and yields 2D [B, D] embedding vectors.
        condition_embedding: Function that takes in conditioning from
            condition_loader and yields 2D [B, D] embedding vectors.
        reference_stats_path: File path to save precomputed statistics of
            reference distribution. Default: current directory.
        save_reference_stats: Boolean indicating whether statistics of
            reference distribution should be saved. Default: False.
        samples_per_condition: Integer indicating the number of samples to
            generate for each condition from the condition_loader. Default: 1.
        cuda: Boolean indicating whether to use GPU accelerated FJD or not.
              Default: False.
        eps: Float value which is added to diagonals of covariance matrices
             to improve computational stability. Default: 1e-6.
    """

    def __init__(self,
                 gan,
                 ref_loader,
                 loader,
                 image_embedding,
                 condition_embedding,
                 cuda=False,
                 args=None,
                 eps=1e-6):

        self.gan = gan
        self.args = args
        self.ref_loader = ref_loader
        self.loader = loader
        self.image_embedding = image_embedding
        self.condition_embedding = condition_embedding
        self.cuda = cuda
        self.eps = eps
        self.gen_embeds, self.cond_embeds, self.true_embeds = None, None, None

        self.mu_fake, self.sigma_fake = None, None
        self.mu_real, self.sigma_real = None, None

        self.transforms = torch.nn.Sequential(
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    def _get_joint_statistics(self, image_embed, cond_embed):
        if self.cuda:
            joint_embed = torch.cat([image_embed, cond_embed], dim=1)
        else:
            joint_embed = np.concatenate([image_embed, cond_embed], axis=1)
        print(joint_embed.shape)
        mu, sigma = get_embedding_statistics(joint_embed, cuda=self.cuda)

        return mu, sigma

    def _calculate_alpha(self, image_embed, cond_embed):
        self.alpha = calculate_alpha(image_embed, cond_embed, cuda=self.cuda)
        return self.alpha

    def _get_embed_im(self, inp, mean, std):
        embed_ims = torch.zeros(size=(inp.size(0), 3, 128, 128),
                                device=self.args.device)
        for i in range(inp.size(0)):
            im = inp[i, :, :, :] * std[i, :, None, None] + mean[i, :, None, None]
            im = (im - torch.min(im)) / (torch.max(im) - torch.min(im))
            embed_ims[i, :, :, :] = im

        return embed_ims

    def _get_embed_im(self, inp):
        embed_ims = torch.zeros(size=(inp.size(0), 3, 384, 384),
                                device=self.args.device)
        for i in range(inp.size(0)):
            im = inp[i]
            im = (im - torch.min(im)) / (torch.max(im) - torch.min(im))

            # im = 2 * (im - torch.min(im)) / (torch.max(im) - torch.min(im)) - 1

            embed_ims[i, 0, :, :] = im
            embed_ims[i, 1, :, :] = im
            embed_ims[i, 2, :, :] = im

        return embed_ims

    def _get_generated_distribution(self):
        image_embed = []
        cond_embed = []
        true_embed = []

        ref_directory = '/storage/fastMRI_brain/data/small_T2_test'
        recon_directory = f'/storage/fastMRI_brain/Langevin_Recons_R=8/'

        for filename in os.listdir(ref_directory):
            for i in range(6):
                recon_object = None
                with torch.no_grad():
                    for j in range(32):
                        try:
                            new_filename = recon_directory + filename + f'|langevin|slide_idx_{i}_R=8_sample={j}_outputs.pt'
                            recon_object = torch.load(new_filename)

                            recon = complex_abs(recon_object['mvue'][0].permute(1, 2, 0)).cuda().unsqueeze(0).unsqueeze(0)
                            true = recon_object['gt'][0][0].abs().unsqueeze(0).unsqueeze(0)
                            zfr = recon_object['zfr'][0].abs().cuda().unsqueeze(0).unsqueeze(0)

                            image = self._get_embed_im(recon)
                            true_im = self._get_embed_im(true)
                            condition_im = self._get_embed_im(zfr)

                            img_e = self.image_embedding(self.transforms(image))
                            true_e = self.image_embedding(self.transforms(true_im))
                            cond_e = self.condition_embedding(self.transforms(condition_im))

                            if self.cuda:
                                image_embed.append(img_e)
                                true_embed.append(true_e)
                                cond_embed.append(cond_e)
                            else:
                                image_embed.append(img_e.cpu().numpy())
                                cond_embed.append(cond_e.cpu().numpy())
                        except KeyboardInterrupt:
                            exit()
                        except Exception as e:
                            print(e)
                            print(recon_directory + filename + f'|langevin|slide_idx_{i}_R=8_sample={j}_outputs.pt')
                            break

        print("WE GOT EMBEDDINGS BABY")
        if self.cuda:
            image_embed = torch.cat(image_embed, dim=0)
            cond_embed = torch.cat(cond_embed, dim=0)
            true_embed = torch.cat(true_embed, dim=0)
        else:
            image_embed = np.concatenate(image_embed, axis=0)
            cond_embed = np.concatenate(cond_embed, axis=0)

        return image_embed.to(dtype=torch.float64), cond_embed.to(dtype=torch.float64), true_embed.to(
            dtype=torch.float64)

    def _get_statistics_from_file(self, path):
        print('Loading reference statistics from {}'.format(path))
        assert path.endswith('.npz'), 'Invalid filepath "{}". Should be .npz'.format(path)

        f = np.load(path)
        mu, sigma, alpha = f['mu'][:], f['sigma'][:], f['alpha']
        f.close()

        if self.cuda:
            mu = torch.tensor(mu).cuda()
            sigma = torch.tensor(sigma).cuda()
            alpha = torch.tensor(alpha).cuda()

        return mu, sigma, alpha

    def _get_reference_distribution(self):
        stats = self._get_statistics_from_file('/storage/fastMRI/ref_stats.npz')
        mu_real, sigma_real, alpha = stats

        self.mu_real, self.sigma_real, self.alpha = mu_real, sigma_real, alpha

        return mu_real, sigma_real

    def get_cfid_torch_pinv(self):
        y_predict, x_true, y_true = self._get_generated_distribution()

        # mean estimations
        y_true = y_true.to(x_true.device)
        m_y_predict = torch.mean(y_predict, dim=0)
        m_x_true = torch.mean(x_true, dim=0)
        m_y_true = torch.mean(y_true, dim=0)

        no_m_y_true = y_true - m_y_true
        no_m_y_pred = y_predict - m_y_predict
        no_m_x_true = x_true - m_x_true

        c_y_predict_x_true = torch.matmul(no_m_y_pred.t(), no_m_x_true) / y_predict.shape[0]
        c_y_predict_y_predict = torch.matmul(no_m_y_pred.t(), no_m_y_pred) / y_predict.shape[0]
        c_x_true_y_predict = torch.matmul(no_m_x_true.t(), no_m_y_pred) / y_predict.shape[0]

        c_y_true_x_true = torch.matmul(no_m_y_true.t(), no_m_x_true) / y_predict.shape[0]
        c_x_true_y_true = torch.matmul(no_m_x_true.t(), no_m_y_true) / y_predict.shape[0]
        c_y_true_y_true = torch.matmul(no_m_y_true.t(), no_m_y_true) / y_predict.shape[0]

        inv_c_x_true_x_true = torch.linalg.pinv(torch.matmul(no_m_x_true.t(), no_m_x_true) / y_predict.shape[0])

        c_y_true_given_x_true = c_y_true_y_true - torch.matmul(c_y_true_x_true,
                                                               torch.matmul(inv_c_x_true_x_true, c_x_true_y_true))
        c_y_predict_given_x_true = c_y_predict_y_predict - torch.matmul(c_y_predict_x_true,
                                                                        torch.matmul(inv_c_x_true_x_true,
                                                                                     c_x_true_y_predict))
        c_y_true_x_true_minus_c_y_predict_x_true = c_y_true_x_true - c_y_predict_x_true
        c_x_true_y_true_minus_c_x_true_y_predict = c_x_true_y_true - c_x_true_y_predict

        # Distance between Gaussians
        m_dist = torch.einsum('...k,...k->...', m_y_true - m_y_predict, m_y_true - m_y_predict)
        c_dist1 = torch.trace(
            torch.matmul(torch.matmul(c_y_true_x_true_minus_c_y_predict_x_true, inv_c_x_true_x_true),
                         c_x_true_y_true_minus_c_x_true_y_predict))
        c_dist_2_1 = torch.trace(c_y_true_given_x_true + c_y_predict_given_x_true)
        c_dist_2_2 = - 2 * trace_sqrt_product_torch(
            c_y_predict_given_x_true, c_y_true_given_x_true)

        c_dist2 = c_dist_2_1 + c_dist_2_2

        cfid = m_dist + c_dist1 + c_dist2

        c_dist = c_dist1 + c_dist2
        print(f"M: {m_dist.cpu().numpy()}")
        print(f"C: {c_dist.cpu().numpy()}")

        return cfid.cpu().numpy()


def get_embedding_statistics(embeddings, cuda=False):
    if cuda:
        embeddings = embeddings.double()  # More precision = more stable
        mu = torch.mean(embeddings, 0)
        sigma = torch_cov(embeddings, rowvar=False)
    else:
        mu = np.mean(embeddings, axis=0)
        sigma = np.cov(embeddings, rowvar=False)
    return mu, sigma


def calculate_alpha(image_embed, cond_embed, cuda=False):
    if cuda:
        image_norm = torch.mean(torch.norm(image_embed, dim=1))
        cond_norm = torch.mean(torch.norm(cond_embed, dim=1))
        alpha = (image_norm / cond_norm).item()
    else:
        image_norm = np.mean(linalg.norm(image_embed, axis=1))
        cond_norm = np.mean(linalg.norm(cond_embed, axis=1))
        alpha = image_norm / cond_norm
    return alpha


def calculate_fd(mu1, sigma1, mu2, sigma2, cuda=False, eps=1e-6):
    if True:
        fid = torch_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=eps)
        fid = fid.cpu().numpy()
    else:
        fid = numpy_calculate_frechet_distance(mu1.cpu().numpy(), sigma1.cpu().numpy(), mu2.cpu().numpy(), sigma2.cpu().numpy(), eps=eps)
    return fid


# A pytorch implementation of cov, from Modar M. Alfadly
# https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def torch_cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.
        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
        Args:
                m: A 1-D or 2-D array containing multiple variables and observations.
                        Each row of `m` represents a variable, and each column a single
                        observation of all those variables.
                rowvar: If `rowvar` is True, then each row represents a
                        variable, with observations in the columns. Otherwise, the
                        relationship is transposed: each column represents a variable,
                        while the rows contain observations.
        Returns:
                The covariance matrix of the variables.
        '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


# Pytorch implementation of matrix sqrt, from Tsung-Yu Lin, and Subhransu Maji
# https://github.com/msubhransu/matrix-sqrt
def sqrt_newton_schulz(A, numIters, dtype=None):
    with torch.no_grad():
        if dtype is None:
            dtype = A.type()
        batchSize = A.shape[0]
        dim = A.shape[1]
        normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
        Y = A.div(normA.view(batchSize, 1, 1).expand_as(A)).to(A.device)
        I = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype).to(A.device)
        Z = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype).to(A.device)
        for i in range(numIters):
            T = 0.5 * (3.0 * I - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)
        sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    return sA


def numpy_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    Taken from https://github.com/bioinf-jku/TTUR
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
                    d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
                         inception net (like returned by the function 'get_predictions')
                         for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
                         representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
                         representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            print('Imaginary component of {}, may affect results'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    out = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return out


# PyTorch implementation of Frechet distance, from Andrew Brock (modified slightly)
# https://github.com/ajbrock/BigGAN-PyTorch/blob/master/inception_utils.py
def torch_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Pytorch implementation of the Frechet Distance.
    Taken from https://github.com/bioinf-jku/TTUR
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
          d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
             inception net (like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
             representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
             representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    # Using double precision instead of float seems to make the GPU FD more stable
    mu1, mu2 = mu1.double(), mu2.double()
    sigma1, sigma2 = sigma1.double(), sigma2.double()

    # Add a tiny offset to the covariance matrices to make covmean estimate more stable
    # Will change the output by a couple decimal places compared to not doing this
    offset = torch.eye(sigma1.size(0)).to(sigma1.device).double() * eps
    sigma1, sigma2 = sigma1 + offset, sigma2 + offset

    diff = mu1 - mu2

    # Run 50 itrs of newton-schulz to get the matrix sqrt of sigma1 dot sigma2
    covmean = sqrt_newton_schulz(sigma1.mm(sigma2).unsqueeze(0), 50).squeeze()
    tr_covmean = torch.trace(covmean)

    out = diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean
    return out