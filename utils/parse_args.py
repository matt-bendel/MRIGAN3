import pathlib

from utils.args import Args


def create_arg_parser():
    # CREATE THE PARSER
    parser = Args()

    # GAN ARGS
    parser.add_argument('--num-iters-discriminator', type=int, default=1,
                        help='Number of iterations of the discriminator')
    parser.add_argument('--num-z', type=int, default=2,
                        help='Number of z values to use during training.')
    parser.add_argument('--latent-size', type=int, default=512, help='Size of latent vector for z location 2')

    # LEARNING ARGS
    parser.add_argument('--batch-size', default=4, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--beta_1', type=float, default=0, help='Beta 1 for Adam')
    parser.add_argument('--beta_2', type=float, default=0.99, help='Beta 2 for Adam')
    parser.add_argument('--adv-weight', type=float, default=1e-3, help='Weight for adversarial loss')
    parser.add_argument('--var-weight', type=float, default=0.01, help='Weight for variance reward')
    parser.add_argument('--ssim-weight', type=float, default=0.84, help='Weight for supervised loss')
    parser.add_argument('--gp-weight', type=float, default=10, help='Weight for Gradient Penalty')

    # DATA ARGS
    parser.add_argument('--R', default=4, type=int,
                        help='Acceleration rate - 4 or 8')
    parser.add_argument('--data-parallel', required=True, action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--num_of_top_slices', default=8, type=int,
                        help='top slices have bigger brain image and less air region')
    parser.add_argument('--use-middle-slices', action='store_true',
                        help='If set, only uses central slice of every data collection')

    # LOGISTICAL ARGS
    parser.add_argument('--dynamic-inpaint', action='store_true',
                        help='Whether or not to dynamically remove chunk of image')
    parser.add_argument('--device', type=int, default=0,
                        help='Which device to train on. Use idx of cuda device or -1 for CPU')
    #TODO UPDATE EXPDIR
    parser.add_argument('--exp-dir', type=pathlib.Path,
                        default=pathlib.Path('/home/bendel.8/Git_Repos/full-scale-mrigan/MRIGAN3/trained_models'),
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint-dir', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')

    # TEST ARGS
    parser.add_argument('--supervised-metrics', action='store_true', help='Evaluate test metrics')
    parser.add_argument('--generative-metrics', action='store_true', help='Evaluate test metrics')
    parser.add_argument('--single-plot', action='store_true', help='Generate plots')

    return parser
