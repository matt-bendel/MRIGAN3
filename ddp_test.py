import glob
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


if __name__ == "__main__":
    cuda = True if torch.cuda.is_available() else False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    args = create_arg_parser().parse_args()
    # restrict visible cuda devices
    # if args.data_parallel or (args.device >= 0):
    #     if not args.data_parallel:
    #         os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    #     args.device = torch.device('cuda')
    # else:
    #     args.device = torch.device('cpu')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    args.in_chans = 16
    args.out_chans = 16

    mp.spawn(
        main,
        args=(args),
        nprocs=world_size
    )
